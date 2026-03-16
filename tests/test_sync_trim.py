from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from footage_studio.core.files import get_created_time
from footage_studio.processing.trim import sync_trim


def dt(h, m, s=0, us=0):
    """Helper: UTC datetime for today at h:m:s."""
    return datetime(2025, 1, 1, h, m, s, us, tzinfo=timezone.utc)


# --- get_created_time: DJI filename fallback ---

def test_get_created_time_dji_filename(tmp_path):
    """Falls back to DJI filename timestamp when no creation_time tag exists."""
    f = tmp_path / "DJI_20250603142537_0001_D.MP4"
    f.touch()
    with patch("footage_studio.core.files.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.probe.return_value = {"format": {"tags": {}}}
        result = get_created_time(f)
    assert result == datetime(2025, 6, 3, 14, 25, 37, tzinfo=timezone.utc)


def test_get_created_time_prefers_metadata_tag(tmp_path):
    """Prefers creation_time from metadata over DJI filename."""
    f = tmp_path / "DJI_20250603000000_0001_D.MP4"
    f.touch()
    with patch("footage_studio.core.files.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.probe.return_value = {
            "format": {"tags": {"creation_time": "2025-06-03T14:25:37.000Z"}}
        }
        result = get_created_time(f)
    assert result == datetime(2025, 6, 3, 14, 25, 37, tzinfo=timezone.utc)


def test_get_created_time_falls_back_to_ctime(tmp_path):
    """Falls back to filesystem ctime when no tag and no DJI filename pattern."""
    f = tmp_path / "some_video.mp4"
    f.touch()
    with patch("footage_studio.core.files.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.probe.return_value = {"format": {"tags": {}}}
        result = get_created_time(f)
    # Just check it returns a timezone-aware datetime
    assert result.tzinfo is not None


# --- sync_trim: time window math ---

def _patch_sync_trim(left_start, left_dur, right_start, right_dur):
    """
    Context managers to patch get_created_time and get_video_duration
    so sync_trim uses the given values.
    Returns (trim_calls,) where trim_calls is populated after sync_trim runs.
    """
    import footage_studio.processing.trim as trim_module

    calls = []

    def fake_trim(filepath, offset, duration, output_path):
        calls.append({"filepath": filepath, "offset": offset, "duration": duration})

    def fake_created_time(fp):
        return left_start if "left" in fp.name else right_start

    def fake_finish_time(fp):
        from datetime import timedelta
        dur = left_dur if "left" in fp.name else right_dur
        return fake_created_time(fp) + timedelta(seconds=dur)

    return calls, fake_created_time, fake_finish_time, fake_trim


def run_sync_trim_mock(left_start, left_dur, right_start, right_dur, tmp_path):
    """Run sync_trim with mocked time functions, return trim call args."""
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    out_l = tmp_path / "out_left.mp4"
    out_r = tmp_path / "out_right.mp4"

    calls, fake_created, fake_finish, fake_trim = _patch_sync_trim(
        left_start, left_dur, right_start, right_dur
    )

    with (
        patch("footage_studio.processing.trim.get_created_time", side_effect=fake_created),
        patch("footage_studio.processing.trim.get_finish_time", side_effect=fake_finish),
        patch("footage_studio.processing.trim.trim", side_effect=fake_trim),
    ):
        sync_trim(left, right, out_l, out_r)

    return calls


def test_sync_trim_identical_start_times(tmp_path):
    """Both cameras start at the same time — no trimming needed (offset=0 for both)."""
    calls = run_sync_trim_mock(
        left_start=dt(10, 0, 0), left_dur=30.0,
        right_start=dt(10, 0, 0), right_dur=30.0,
        tmp_path=tmp_path,
    )
    left_trim, right_trim = calls[0], calls[1]
    assert left_trim["offset"] == pytest.approx(0.0)
    assert right_trim["offset"] == pytest.approx(0.0)
    assert left_trim["duration"] == pytest.approx(30.0)
    assert right_trim["duration"] == pytest.approx(30.0)


def test_sync_trim_left_starts_later(tmp_path):
    """Left camera starts 4s after right — right is trimmed by 4s offset."""
    calls = run_sync_trim_mock(
        left_start=dt(10, 0, 4), left_dur=40.0,   # ends at 10:00:44
        right_start=dt(10, 0, 0), right_dur=40.0,  # ends at 10:00:40
        tmp_path=tmp_path,
    )
    left_trim = next(c for c in calls if "left" in c["filepath"].name)
    right_trim = next(c for c in calls if "right" in c["filepath"].name)

    # shared window: [10:00:04, 10:00:40] = 36s
    assert left_trim["offset"] == pytest.approx(0.0)
    assert right_trim["offset"] == pytest.approx(4.0)
    assert left_trim["duration"] == pytest.approx(36.0)
    assert right_trim["duration"] == pytest.approx(36.0)


def test_sync_trim_right_starts_later(tmp_path):
    """Right camera starts 4s after left — left is trimmed by 4s offset."""
    calls = run_sync_trim_mock(
        left_start=dt(10, 0, 0), left_dur=40.0,   # ends at 10:00:40
        right_start=dt(10, 0, 4), right_dur=40.0,  # ends at 10:00:44
        tmp_path=tmp_path,
    )
    left_trim = next(c for c in calls if "left" in c["filepath"].name)
    right_trim = next(c for c in calls if "right" in c["filepath"].name)

    # shared window: [10:00:04, 10:00:40] = 36s
    assert left_trim["offset"] == pytest.approx(4.0)
    assert right_trim["offset"] == pytest.approx(0.0)
    assert left_trim["duration"] == pytest.approx(36.0)
    assert right_trim["duration"] == pytest.approx(36.0)


def test_sync_trim_different_end_times(tmp_path):
    """Cameras start together but right ends 5s earlier — shared window is shorter."""
    calls = run_sync_trim_mock(
        left_start=dt(10, 0, 0), left_dur=40.0,   # ends at 10:00:40
        right_start=dt(10, 0, 0), right_dur=35.0,  # ends at 10:00:35
        tmp_path=tmp_path,
    )
    left_trim = next(c for c in calls if "left" in c["filepath"].name)
    right_trim = next(c for c in calls if "right" in c["filepath"].name)

    # shared window: [10:00:00, 10:00:35] = 35s
    assert left_trim["offset"] == pytest.approx(0.0)
    assert right_trim["offset"] == pytest.approx(0.0)
    assert left_trim["duration"] == pytest.approx(35.0)
    assert right_trim["duration"] == pytest.approx(35.0)


def test_sync_trim_no_overlap_raises(tmp_path):
    """Videos with no overlapping time window raise ValueError."""
    with pytest.raises(ValueError, match="no overlapping time window"):
        run_sync_trim_mock(
            left_start=dt(10, 0, 0), left_dur=10.0,   # ends at 10:00:10
            right_start=dt(10, 0, 20), right_dur=10.0,  # starts at 10:00:20
            tmp_path=tmp_path,
        )


def test_sync_trim_dji_real_world_scenario(tmp_path):
    """
    Mirrors the actual DJI test footage:
    left starts at 17:40:31, right at 17:40:27 (right 4s earlier),
    both ~40s long → shared window is ~36.5s, right offset 4s.
    """
    calls = run_sync_trim_mock(
        left_start=dt(17, 40, 31), left_dur=40.256,
        right_start=dt(17, 40, 27), right_dur=40.555,
        tmp_path=tmp_path,
    )
    left_trim = next(c for c in calls if "left" in c["filepath"].name)
    right_trim = next(c for c in calls if "right" in c["filepath"].name)

    # shared_start = 17:40:31, shared_end = 17:41:07.555
    # duration = 36.555s, left_offset = 0, right_offset = 4s
    assert left_trim["offset"] == pytest.approx(0.0)
    assert right_trim["offset"] == pytest.approx(4.0)
    assert left_trim["duration"] == pytest.approx(36.555, abs=0.01)
    assert right_trim["duration"] == pytest.approx(36.555, abs=0.01)
