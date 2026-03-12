import os
import subprocess
import tempfile
from pathlib import Path

import ffmpeg


def set_metadata(filepath: Path, key: str, value: str) -> None:
    """Write a metadata key/value to a video file using ffmpeg stream copy."""
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=filepath.parent)
    tmp_path = Path(tmp_name)
    try:
        os.close(tmp_fd)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(filepath),
                "-c",
                "copy",
                "-metadata",
                f"{key}={value}",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        os.replace(str(tmp_path), str(filepath))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def get_metadata(filepath: Path, key: str) -> str | None:
    """Read a metadata value from a video file."""
    probe = ffmpeg.probe(str(filepath))
    tags = probe.get("format", {}).get("tags", {})
    tags_lower = {k.lower(): v for k, v in tags.items()}
    return tags_lower.get(key.lower())


def get_files_by_status(
    directory: Path, status: str, recursive: bool = False
) -> list[Path]:
    """Return all MP4 files in directory with the given status metadata value."""
    pattern = "**/*.mp4" if recursive else "*.mp4"
    results = []
    for filepath in sorted(directory.glob(pattern)):
        if get_metadata(filepath, "status") == status:
            results.append(filepath)
    return results
