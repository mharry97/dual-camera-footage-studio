from dataclasses import dataclass
from pathlib import Path

from footage_studio.core import get_created_time, get_files_by_status


@dataclass
class Session:
    name: str
    left: Path
    right: Path


def scan_sessions(footage_dir: Path) -> list[Session]:
    """Match GROUPED left and right camera files into sessions by created time (within 60s)."""
    left_files = get_files_by_status(footage_dir / "Left Camera", "GROUPED")
    right_files = get_files_by_status(footage_dir / "Right Camera", "GROUPED")

    right_with_times = [(get_created_time(fp), fp) for fp in right_files]

    sessions = []
    for left_fp in left_files:
        left_created = get_created_time(left_fp)
        for right_created, right_fp in right_with_times:
            if abs((left_created - right_created).total_seconds()) <= 60:
                sessions.append(Session(
                    name=left_fp.stem,
                    left=left_fp,
                    right=right_fp,
                ))
                break

    return sessions
