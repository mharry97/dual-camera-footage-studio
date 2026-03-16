from dataclasses import dataclass
from pathlib import Path

from footage_studio.core import get_created_time, get_files_by_status, get_files_without_status


@dataclass
class Session:
    name: str
    left: Path
    right: Path


def scan_sessions(left_dir: Path, right_dir: Path) -> list[Session]:
    """Match left and right camera files (GROUPED or ungrouped) into sessions by created time (within 60s)."""
    left_files = (
        get_files_by_status(left_dir, "GROUPED")
        + get_files_without_status(left_dir)
    )
    right_files = (
        get_files_by_status(right_dir, "GROUPED")
        + get_files_without_status(right_dir)
    )

    left_with_times = [(get_created_time(fp), fp) for fp in left_files]
    right_with_times = [(get_created_time(fp), fp) for fp in right_files]

    candidates = []
    for left_created, left_fp in left_with_times:
        for right_created, right_fp in right_with_times:
            gap = abs((left_created - right_created).total_seconds())
            if gap <= 60:
                candidates.append((gap, left_created, left_fp, right_fp))

    candidates.sort(key=lambda x: x[0])

    used_left: set[Path] = set()
    used_right: set[Path] = set()
    sessions = []
    for _, left_created, left_fp, right_fp in candidates:
        if left_fp in used_left or right_fp in used_right:
            continue
        used_left.add(left_fp)
        used_right.add(right_fp)
        sessions.append(Session(name=left_fp.stem, left=left_fp, right=right_fp))

    sessions.sort(key=lambda s: s.left)
    return sessions
