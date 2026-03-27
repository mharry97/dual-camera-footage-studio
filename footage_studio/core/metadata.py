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
                "-movflags",
                "use_metadata_tags",
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


def glob_mp4(directory: Path, recursive: bool = False) -> list[Path]:
    """Return all .mp4/.MP4 files in directory, sorted by name. Excludes hidden directories."""
    prefix = "**/" if recursive else ""
    files = set(directory.glob(f"{prefix}*.mp4")) | set(directory.glob(f"{prefix}*.MP4"))
    return sorted(f for f in files if not any(part.startswith(".") for part in f.relative_to(directory).parts))


def get_files_by_status(
    directory: Path, status: str, recursive: bool = False
) -> list[Path]:
    """Return all MP4 files in directory with the given status metadata value."""
    results = []
    for filepath in glob_mp4(directory, recursive=recursive):
        if get_metadata(filepath, "status") == status:
            results.append(filepath)
    return results


def get_files_without_status(directory: Path, recursive: bool = False) -> list[Path]:
    """Return all MP4 files in directory that have no status metadata tag."""
    results = []
    for filepath in glob_mp4(directory, recursive=recursive):
        if get_metadata(filepath, "status") is None:
            results.append(filepath)
    return results
