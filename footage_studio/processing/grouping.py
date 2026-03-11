from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from footage_studio.core import get_created_time, get_metadata, get_video_duration


@dataclass
class FileInfo:
    path: Path
    name: str
    created: datetime
    duration: float  # seconds


@dataclass
class Group:
    name: str  # formatted created time of first file: "yyyy-mm-dd hh:mm"
    files: list[FileInfo] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(f.duration for f in self.files)

    @property
    def output_name(self) -> str:
        """Filename-safe version of the group name."""
        return self.files[0].created.strftime("%Y-%m-%d_%H-%M") + ".mp4"


def scan_camera_dir(camera_dir: Path) -> list[Group]:
    """Scan a camera directory and group consecutive clips by recording session."""
    unprocessed = []
    for fp in sorted(camera_dir.glob("*.mp4")):
        if get_metadata(fp, "status") != "PROCESSED":
            created = get_created_time(fp)
            duration = get_video_duration(fp)
            unprocessed.append(FileInfo(fp, fp.name, created, duration))

    if not unprocessed:
        return []

    unprocessed.sort(key=lambda f: f.created)

    groups: list[Group] = []
    current = Group(
        name=unprocessed[0].created.strftime("%Y-%m-%d %H:%M"),
        files=[unprocessed[0]],
    )

    for fi in unprocessed[1:]:
        prev = current.files[-1]
        finish = prev.created + timedelta(seconds=prev.duration)
        if abs((fi.created - finish).total_seconds()) <= 1:
            current.files.append(fi)
        else:
            groups.append(current)
            current = Group(name=fi.created.strftime("%Y-%m-%d %H:%M"), files=[fi])

    groups.append(current)
    return groups
