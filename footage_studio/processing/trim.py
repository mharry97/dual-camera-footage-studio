import json
import subprocess
import tempfile
from pathlib import Path

from footage_studio.core import get_created_time, get_finish_time, get_video_duration
from footage_studio.processing.audio_sync import CONFIDENCE_THRESHOLD, audio_offset

# Snap tolerance: offsets within half a frame (at 30fps) of a keyframe are
# treated as keyframe-aligned, allowing a pure stream copy.
_KEYFRAME_SNAP_S = 1.0 / 60.0


def head_edit_list_s(filepath: Path) -> float:
    """
    Return the head trim (seconds) encoded as an mp4 edit list, 0.0 if none.

    Files with a large head edit list play differently across decoders: ffmpeg
    honours the trim, but some browsers (notably Firefox) show the pre-trim
    frames, shifting the whole timeline. Anything produced by this tool should
    keep this at (or within a couple of frames of) zero.
    """
    trace = subprocess.run(
        ["ffprobe", "-v", "trace", "-i", str(filepath)],
        capture_output=True,
    ).stderr.decode(errors="replace")
    for line in trace.splitlines():
        if "edit list 0 - media time:" in line:
            try:
                media_time = int(line.split("media time:")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                continue
            timescale = _video_timescale(filepath)
            return media_time / timescale if timescale else 0.0
    return 0.0


def _video_timescale(filepath: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=time_base", "-of", "json", str(filepath),
        ],
        capture_output=True,
    )
    try:
        time_base = json.loads(result.stdout)["streams"][0]["time_base"]
        return int(time_base.split("/")[1])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return 0


def _first_keyframe_at_or_after(filepath: Path, offset: float) -> float | None:
    """Raw-media PTS (seconds, edit list ignored) of the first keyframe at or after offset."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-ignore_editlist", "1",
            "-select_streams", "v:0",
            "-skip_frame", "nokey",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            "-read_intervals", f"{max(0.0, offset - 0.5)}%+#600",
            str(filepath),
        ],
        capture_output=True,
    )
    for line in result.stdout.decode().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            pts = float(line)
        except ValueError:
            continue
        if pts >= offset - _KEYFRAME_SNAP_S:
            return pts
    return None


def _probe_encode_params(filepath: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,pix_fmt,profile,level,r_frame_rate",
            "-of", "json", str(filepath),
        ],
        capture_output=True,
    )
    return json.loads(result.stdout)["streams"][0]


def trim(filepath: Path, offset: float, duration: float, output_path: Path) -> None:
    """
    Trim a video to [offset, offset + duration] without leaving an mp4 edit list.

    A plain `ffmpeg -ss <offset> -c copy` cannot cut mid-GOP, so ffmpeg keeps the
    packets back to the previous keyframe and represents the intended start as an
    edit list — which ffmpeg-family players honour but some browsers ignore,
    producing files whose timeline differs per player.

    Instead this smart-cuts: the head (offset → next keyframe) is re-encoded
    frame-accurately, the remainder is stream-copied from the keyframe, and the
    two parts are concatenated. Cost is one GOP of re-encoding (seconds), not a
    full transcode. If offset already sits on a keyframe the head part is skipped
    and this degrades to a pure stream copy.

    Video-only (studio outputs carry no audio).
    """
    # All cutting happens on the raw media timeline (-ignore_editlist): seeking
    # an elst-carrying source lands unpredictably (the demuxer can pick the
    # keyframe a GOP early and mask it with a fresh edit list, which a later
    # concat then discards). The caller's offset is expressed on the default
    # (edit-list-applied) timeline, so translate it first.
    elst_s = head_edit_list_s(filepath)
    raw_offset = offset + elst_s

    keyframe = _first_keyframe_at_or_after(filepath, raw_offset)
    if keyframe is None:
        raise RuntimeError(f"No keyframe found after offset {offset:.3f}s in {filepath}")

    raw_end = raw_offset + duration
    ignore_elst = ["-ignore_editlist", "1"]

    if keyframe - raw_offset <= _KEYFRAME_SNAP_S:
        # Keyframe-aligned: input seek on the raw timeline, no edit list needed.
        _run_ffmpeg([
            *ignore_elst,
            "-ss", f"{keyframe + 0.005:.6f}", "-i", str(filepath),
            "-t", f"{max(0.0, raw_end - keyframe):.6f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path),
        ])
        return

    if keyframe >= raw_end:
        # Entire requested range sits inside one GOP: re-encode all of it.
        _run_ffmpeg([
            *ignore_elst,
            "-ss", f"{raw_offset:.6f}", "-t", f"{duration:.6f}", "-i", str(filepath),
            *_head_encode_args(filepath),
            "-movflags", "+faststart",
            str(output_path),
        ])
        return

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        head = tmp_dir / "head.mp4"
        tail = tmp_dir / "tail.mp4"

        # Head: frame-accurate re-encode from raw_offset up to the keyframe.
        _run_ffmpeg([
            *ignore_elst,
            "-ss", f"{raw_offset:.6f}", "-to", f"{keyframe:.6f}", "-i", str(filepath),
            *_head_encode_args(filepath),
            str(head),
        ])

        # Tail: pure stream copy starting exactly on the keyframe. The seek
        # target sits a few ms past the keyframe PTS so float rounding can
        # never select the previous keyframe (input seek picks pts <= target).
        _run_ffmpeg([
            *ignore_elst,
            "-ss", f"{keyframe + 0.005:.6f}", "-i", str(filepath),
            "-t", f"{max(0.0, raw_end - keyframe):.6f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(tail),
        ])

        concat_list = tmp_dir / "concat.txt"
        concat_list.write_text(f"file '{head.resolve()}'\nfile '{tail.resolve()}'\n")
        _run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-movflags", "+faststart",
            str(output_path),
        ])


def _head_encode_args(filepath: Path) -> list[str]:
    """x264 args matching the source stream closely enough for clean concat."""
    params = _probe_encode_params(filepath)
    timescale = _video_timescale(filepath) or 30000
    args = [
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", params.get("pix_fmt", "yuv420p"),
        "-video_track_timescale", str(timescale),
        "-an",
    ]
    profile = (params.get("profile") or "").lower()
    if profile in ("baseline", "main", "high"):
        args += ["-profile:v", profile]
    return args


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {stderr[-2000:]}")


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
    """Trim two videos to their shared time window (timestamp-based).

    Uses the smart-cut trim above, so outputs carry no edit list and play
    identically in every player.
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
