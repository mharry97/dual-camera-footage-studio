from pathlib import Path

SUBDIRECTORIES = ["Left Camera", "Right Camera", "Output Footage"]


def check_subdirectories(base_dir: Path) -> dict[str, bool]:
    """Return which expected subdirectories exist in base_dir."""
    return {name: (base_dir / name).is_dir() for name in SUBDIRECTORIES}


def get_total_size(filepaths: list[Path]) -> int:
    """Return total size of files in bytes."""
    return sum(fp.stat().st_size for fp in filepaths)
