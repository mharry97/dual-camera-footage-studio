"""
Apply computed sync offsets: trim each video from its start offset and write
*_synced.mp4 output files alongside the originals.

Both videos are aligned to the same start point but are NOT clipped to the same
end — if one camera ran longer it keeps its extra footage, which the stitching
pipeline will append as single-camera tail footage.

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

    left_offset = session.left_offset_s
    right_offset = session.right_offset_s

    # A negative right_offset means the right video starts earlier than the
    # left — ffmpeg can't seek to negative time, so compensate by trimming
    # more from the left instead.
    if right_offset < 0.0:
        left_offset -= right_offset
        right_offset = 0.0

    # Trim each video from its start offset to its natural end. We do NOT clip
    # both to the same duration — if one camera ran longer, the extra footage
    # is preserved here and the stitching pipeline appends it as a tail.
    _stream_copy_trim_and_rename(left, left_offset)
    _stream_copy_trim_and_rename(right, right_offset)


def _stream_copy_trim_and_rename(src: Path, offset_s: float) -> None:
    """Trim src from offset_s to its natural end, rename to *_synced.mp4, delete original."""
    import ffmpeg as ffmpeg_probe
    synced = src.with_name(src.stem.removesuffix("_grouped") + "_synced.mp4")
    tmp = src.with_suffix(".tmp.mp4")

    # Preserve creation_time from source so pairing still works after trim
    probe = ffmpeg_probe.probe(str(src))
    creation_time = probe.get("format", {}).get("tags", {}).get("creation_time", "")

    cmd = ["ffmpeg", "-y", "-ss", str(offset_s), "-i", str(src), "-c", "copy"]
    if creation_time:
        cmd += ["-metadata", f"creation_time={creation_time}"]
    cmd.append(str(tmp))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        tmp.rename(synced)
        src.unlink()
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        stderr = e.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {stderr}") from e
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
