from pathlib import Path


def glob_mp4(directory: Path, recursive: bool = False) -> list[Path]:
    """Return all .mp4/.MP4 files in directory, sorted by name. Excludes hidden directories."""
    prefix = "**/" if recursive else ""
    files = set(directory.glob(f"{prefix}*.mp4")) | set(directory.glob(f"{prefix}*.MP4"))
    return sorted(f for f in files if not any(part.startswith(".") for part in f.relative_to(directory).parts))
