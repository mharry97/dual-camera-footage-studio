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
    log_path: Path | None = None,
) -> tuple[float, float, float]:
    """
    Compute sync trim offsets for a left/right video pair.

    Returns (left_offset_s, right_offset_s, duration_s) — the number of seconds
    to skip from the start of each video, and the duration of the shared window.

    Uses audio cross-correlation for precision, falling back to creation timestamps.
    If log_path is given, writes a human-readable sync debug log there.
    """
    left_start = get_created_time(left)
    right_start = get_created_time(right)
    left_end = get_finish_time(left)
    right_end = get_finish_time(right)

    shared_start = max(left_start, right_start)
    shared_end = min(left_end, right_end)

    if shared_end <= shared_start:
        raise ValueError("Videos have no overlapping time window.")

    timestamp_offset_seconds = (left_start - right_start).total_seconds()
    timestamp_duration = (shared_end - shared_start).total_seconds()

    log_lines = [
        f"Left:  {left.name}",
        f"Right: {right.name}",
        f"",
        f"Timestamp-based estimate:",
        f"  left start:   {left_start.isoformat()}",
        f"  right start:  {right_start.isoformat()}",
        f"  offset (right - left): {-timestamp_offset_seconds:+.3f}s",
        f"  shared duration: {timestamp_duration:.3f}s",
        f"",
    ]

    audio_result = audio_offset(left, right, search_window_seconds=30.0)

    if audio_result is not None:
        offset_seconds, confidence = audio_result
        log_lines += [
            f"Audio cross-correlation:",
            f"  offset (right - left): {offset_seconds:+.3f}s",
            f"  confidence: {confidence:.4f} (threshold: {CONFIDENCE_THRESHOLD})",
        ]
        if confidence >= CONFIDENCE_THRESHOLD:
            left_trim = max(0.0, -offset_seconds)
            right_trim = max(0.0, offset_seconds)
            duration = min(
                get_video_duration(left) - left_trim,
                get_video_duration(right) - right_trim,
            )
            method = "audio"
            log_lines += [f"  result: USED (confidence above threshold)", f""]
        else:
            log_lines += [f"  result: REJECTED (confidence below threshold)", f""]
            method = "timestamp"
    else:
        log_lines += [
            f"Audio cross-correlation: UNAVAILABLE (no usable audio in one or both files)",
            f"",
        ]
        method = "timestamp"

    if method == "timestamp":
        left_trim = (shared_start - left_start).total_seconds()
        right_trim = (shared_start - right_start).total_seconds()
        duration = timestamp_duration

    log_lines += [
        f"Method used: {method}",
        f"  left trim offset:  {left_trim:.3f}s",
        f"  right trim offset: {right_trim:.3f}s",
        f"  output duration:   {duration:.3f}s",
    ]

    if log_path is not None:
        log_path.write_text("\n".join(log_lines) + "\n")

    return left_trim, right_trim, duration


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
