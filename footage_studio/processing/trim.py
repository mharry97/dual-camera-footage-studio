import subprocess
from pathlib import Path

from footage_studio.core import get_created_time, get_finish_time, get_video_duration
from footage_studio.processing.audio_sync import CONFIDENCE_THRESHOLD, audio_offset


def trim(filepath: Path, offset: float, duration: float, output_path: Path) -> None:
    """Trim a video file to a given offset and duration using ffmpeg stream copy."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(filepath),
            "-ss", str(offset),
            "-t", str(duration),
            "-c", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def compute_sync_offsets(
    left: Path,
    right: Path,
) -> tuple[float, float, float, float | None]:
    """
    Compute sync trim offsets for a left/right video pair.

    Returns (left_offset_s, right_offset_s, duration_s, confidence).
    Uses audio cross-correlation for precision, falling back to creation timestamps.
    """
    left_start = get_created_time(left)
    right_start = get_created_time(right)
    left_end = get_finish_time(left)
    right_end = get_finish_time(right)

    shared_start = max(left_start, right_start)
    shared_end = min(left_end, right_end)

    if shared_end <= shared_start:
        raise ValueError("Videos have no overlapping time window.")

    audio_result = audio_offset(left, right, search_window_seconds=30.0)
    confidence: float | None = audio_result[1] if audio_result is not None else None

    if confidence is not None and confidence >= CONFIDENCE_THRESHOLD:
        offset_seconds = audio_result[0]  # type: ignore[index]
        left_trim = max(0.0, -offset_seconds)
        right_trim = max(0.0, offset_seconds)
        duration = min(
            get_video_duration(left) - left_trim,
            get_video_duration(right) - right_trim,
        )
    else:
        left_trim = (shared_start - left_start).total_seconds()
        right_trim = (shared_start - right_start).total_seconds()
        duration = (shared_end - shared_start).total_seconds()

    return left_trim, right_trim, duration, confidence


def sync_trim(left: Path, right: Path, left_output: Path, right_output: Path) -> None:
    """Trim two videos to their shared time window (timestamp-based, stream copy).

    Note: stream copy is only keyframe-accurate. For sub-keyframe sync precision
    use compute_sync_offsets and seek OpenCV captures directly.
    """
    left_start = get_created_time(left)
    right_start = get_created_time(right)
    left_end = get_finish_time(left)
    right_end = get_finish_time(right)

    shared_start = max(left_start, right_start)
    shared_end = min(left_end, right_end)

    if shared_end <= shared_start:
        raise ValueError("Videos have no overlapping time window.")

    duration = (shared_end - shared_start).total_seconds()
    trim(left, (shared_start - left_start).total_seconds(), duration, left_output)
    trim(right, (shared_start - right_start).total_seconds(), duration, right_output)
