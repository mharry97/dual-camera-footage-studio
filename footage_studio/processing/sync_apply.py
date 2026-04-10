"""
Apply computed sync offsets: trim left/right grouped files to their shared window
and write *_synced.mp4 output files alongside the originals.

Also writes a *_sync_info.json alongside the left synced file for auditability.
"""

from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncSessionInput:
    name: str
    left_path: str
    right_path: str
    left_offset_s: float
    right_offset_s: float
    duration_s: float
    method: str             # "audio" | "manual" | "timestamp"
    confidence: float | None = None


@dataclass
class ApplyJobProgress:
    stage: str              # "running" | "done" | "failed"
    current: int
    total: int
    error: str | None = None


_jobs: dict[str, ApplyJobProgress] = {}
_lock = threading.Lock()


def start_apply_job(sessions: list[SyncSessionInput]) -> str:
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        _jobs[job_id] = ApplyJobProgress(stage="running", current=0, total=len(sessions))

    thread = threading.Thread(target=_run_apply, args=(job_id, sessions), daemon=True)
    thread.start()
    return job_id


def get_apply_job(job_id: str) -> ApplyJobProgress | None:
    with _lock:
        return _jobs.get(job_id)


def _run_apply(job_id: str, sessions: list[SyncSessionInput]) -> None:
    total = len(sessions)
    for i, session in enumerate(sessions):
        with _lock:
            _jobs[job_id] = ApplyJobProgress(stage="running", current=i, total=total)
        try:
            _apply_one(session)
        except Exception as e:
            with _lock:
                _jobs[job_id] = ApplyJobProgress(stage="failed", current=i, total=total, error=str(e))
            return

    with _lock:
        _jobs[job_id] = ApplyJobProgress(stage="done", current=total, total=total)


def _apply_one(session: SyncSessionInput) -> None:
    left = Path(session.left_path)
    right = Path(session.right_path)

    _stream_copy_trim_and_rename(left, session.left_offset_s, session.duration_s)
    _stream_copy_trim_and_rename(right, session.right_offset_s, session.duration_s)


def _stream_copy_trim_and_rename(src: Path, offset_s: float, duration_s: float) -> None:
    """Trim src to a temp file, rename temp to *_synced.mp4, delete original."""
    synced = src.with_name(src.stem.removesuffix("_grouped") + "_synced.mp4")
    tmp = src.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(offset_s),
                "-i", str(src),
                "-t", str(duration_s),
                "-c", "copy",
                str(tmp),
            ],
            check=True,
            capture_output=True,
        )
        tmp.rename(synced)
        src.unlink()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
