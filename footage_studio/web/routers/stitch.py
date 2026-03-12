from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footage_studio.core import check_subdirectories, get_footage_dir
from footage_studio.processing import scan_sessions

router = APIRouter(prefix="/api/stitch")


@router.get("/scan")
async def scan():
    footage_dir = get_footage_dir()
    if not footage_dir:
        return JSONResponse({"status": "no_directory"})

    if not footage_dir.exists():
        return JSONResponse({"status": "directory_not_found"})

    subdirs = check_subdirectories(footage_dir)
    missing = [name for name, exists in subdirs.items() if not exists]
    if missing:
        return JSONResponse({"status": "missing_dirs", "missing": missing})

    sessions = scan_sessions(footage_dir)

    return {
        "status": "ok",
        "sessions": [
            {
                "name": s.name,
                "left": s.left.name,
                "right": s.right.name,
            }
            for s in sessions
        ],
    }
