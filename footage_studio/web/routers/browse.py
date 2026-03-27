from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footage_studio.core import get_output_dir, glob_mp4

router = APIRouter(prefix="/api/browse")


@router.get("/scan")
async def scan():
    output_dir = get_output_dir()
    if not output_dir:
        return JSONResponse({"status": "no_directory"})

    if not output_dir.exists():
        return JSONResponse({"status": "directory_not_found"})

    files = sorted(glob_mp4(output_dir, recursive=True), key=lambda f: f.stat().st_mtime, reverse=True)

    return {
        "status": "ok",
        "files": [{"name": f.stem, "path": str(f.relative_to(output_dir))} for f in files],
    }
