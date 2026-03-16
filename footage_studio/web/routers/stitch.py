from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from footage_studio.core import get_left_camera_dir, get_output_dir, get_right_camera_dir
from footage_studio.processing import scan_sessions
from footage_studio.stitching.jobs import StitchSession, get_job, start_stitch_job

router = APIRouter(prefix="/api/stitch")


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

    sessions = scan_sessions(left_dir, right_dir)

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


class ConfirmSession(BaseModel):
    name: str
    left_path: str
    right_path: str


class ConfirmRequest(BaseModel):
    sessions: list[ConfirmSession]


@router.post("/confirm")
async def confirm(body: ConfirmRequest):
    output_dir = get_output_dir()
    if not output_dir:
        return JSONResponse({"status": "no_directory"}, status_code=400)

    sessions = [
        StitchSession(name=s.name, left_path=s.left_path, right_path=s.right_path)
        for s in body.sessions
    ]
    job_id = start_stitch_job(sessions, output_dir)
    return {"status": "ok", "job_id": job_id}


@router.get("/status/{job_id}")
async def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return JSONResponse({"status": "not_found"}, status_code=404)

    p = job.progress
    return {
        "stage": p.stage,
        "session_index": p.session_index,
        "total_sessions": p.total_sessions,
        "current_frame": p.current_frame,
        "total_frames": p.total_frames,
        "eta_seconds": p.eta_seconds,
        "warnings": p.warnings,
        "error": p.error,
    }
