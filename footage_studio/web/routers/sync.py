from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from footage_studio.core import get_left_camera_dir, get_right_camera_dir
from footage_studio.core.files import get_created_time, get_finish_time, get_video_duration
from footage_studio.processing import scan_sessions
from footage_studio.processing.audio_sync import CONFIDENCE_THRESHOLD, audio_offset
from footage_studio.processing.sync_apply import (
    SyncSessionInput,
    get_apply_job,
    start_apply_job,
)
from footage_studio.stitching.video_io import probe_video, read_frame_at

router = APIRouter(prefix="/api/sync")


@router.get("/scan")
async def scan():
    left_dir = get_left_camera_dir()
    right_dir = get_right_camera_dir()
    if not left_dir or not right_dir:
        return JSONResponse({"status": "no_directory"})
    if not left_dir.exists():
        return JSONResponse({"status": "directory_not_found", "missing": str(left_dir)})
    if not right_dir.exists():
        return JSONResponse({"status": "directory_not_found", "missing": str(right_dir)})

    sessions = scan_sessions(left_dir, right_dir, name_filter="_grouped")
    return {
        "status": "ok",
        "sessions": [
            {
                "name": s.name,
                "left": s.left.name,
                "right": s.right.name,
                "left_path": str(s.left),
                "right_path": str(s.right),
            }
            for s in sessions
        ],
    }


@router.get("/check")
async def check(left_path: str, right_path: str):
    """
    Run audio cross-correlation on a pair and return sync offsets + confidence.
    Falls back to timestamps if audio confidence is below threshold.
    """
    left = Path(left_path)
    right = Path(right_path)

    if not left.exists() or not right.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)

    audio_result = audio_offset(left, right, search_window_seconds=30.0)
    confidence: float | None = audio_result[1] if audio_result is not None else None
    accepted = confidence is not None and confidence >= CONFIDENCE_THRESHOLD

    if accepted:
        offset_s = audio_result[0]  # type: ignore[index]
        left_offset_s = max(0.0, -offset_s)
        right_offset_s = max(0.0, offset_s)
        duration_s = min(
            get_video_duration(left) - left_offset_s,
            get_video_duration(right) - right_offset_s,
        )
        method = "audio"
    else:
        left_start = get_created_time(left)
        right_start = get_created_time(right)
        shared_start = max(left_start, right_start)
        shared_end = min(get_finish_time(left), get_finish_time(right))
        left_offset_s = (shared_start - left_start).total_seconds()
        right_offset_s = (shared_start - right_start).total_seconds()
        duration_s = (shared_end - shared_start).total_seconds()
        method = "timestamp"

    return {
        "status": "ok",
        "confidence": confidence,
        "accepted": accepted,
        "left_offset_s": left_offset_s,
        "right_offset_s": right_offset_s,
        "duration_s": duration_s,
        "method": method,
    }


@router.get("/frame")
async def frame(video_path: str, time_s: float, max_width: int = 640):
    """Return a JPEG-encoded frame from a video at the given timestamp."""
    path = Path(video_path)
    if not path.exists() or path.suffix.lower() not in (".mp4", ".mov", ".avi"):
        return JSONResponse({"status": "not_found"}, status_code=404)

    frame_w, frame_h, _, _ = probe_video(path)
    img = read_frame_at(path, time_s, frame_w, frame_h)
    if img is None:
        return JSONResponse({"status": "error"}, status_code=500)

    if frame_w > max_width:
        scale = max_width / frame_w
        img = cv2.resize(img, (max_width, int(frame_h * scale)))

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


class ApplySession(BaseModel):
    name: str
    left_path: str
    right_path: str
    left_offset_s: float
    right_offset_s: float
    duration_s: float
    method: str
    confidence: float | None = None


class ApplyRequest(BaseModel):
    sessions: list[ApplySession]


@router.post("/apply")
async def apply(body: ApplyRequest):
    sessions = [
        SyncSessionInput(
            name=s.name,
            left_path=s.left_path,
            right_path=s.right_path,
            left_offset_s=s.left_offset_s,
            right_offset_s=s.right_offset_s,
            duration_s=s.duration_s,
            method=s.method,
            confidence=s.confidence,
        )
        for s in body.sessions
    ]
    job_id = start_apply_job(sessions)
    return {"status": "ok", "job_id": job_id}


@router.get("/apply-status/{job_id}")
async def apply_status(job_id: str):
    job = get_apply_job(job_id)
    if job is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {
        "stage": job.stage,
        "current": job.current,
        "total": job.total,
        "error": job.error,
    }
