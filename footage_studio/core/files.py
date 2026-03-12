from datetime import datetime, timedelta, timezone
from pathlib import Path

import ffmpeg


def get_video_duration(filepath: Path) -> float:
    """Return video duration in seconds."""
    probe = ffmpeg.probe(str(filepath))
    return float(probe["format"]["duration"])


def get_created_time(filepath: Path) -> datetime:
    """Return the creation datetime from video metadata, falling back to filesystem ctime."""
    probe = ffmpeg.probe(str(filepath))
    tags = probe.get("format", {}).get("tags", {})
    tags_lower = {k.lower(): v for k, v in tags.items()}
    creation_time = tags_lower.get("creation_time")
    if creation_time:
        return datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    return datetime.fromtimestamp(filepath.stat().st_ctime, tz=timezone.utc)


def get_finish_time(filepath: Path) -> datetime:
    """Return the inferred finish time (created time + duration)."""
    return get_created_time(filepath) + timedelta(seconds=get_video_duration(filepath))
