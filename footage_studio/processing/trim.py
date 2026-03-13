import subprocess
from pathlib import Path

from footage_studio.core import get_created_time, get_finish_time


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


def sync_trim(left: Path, right: Path, left_output: Path, right_output: Path) -> None:
    """Trim two videos to their shared time window."""
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
