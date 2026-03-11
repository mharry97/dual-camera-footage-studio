from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footage_studio.core import check_subdirectories, get_footage_dir

router = APIRouter(prefix="/api/browse")


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

    output_dir = footage_dir / "Output Footage"
    files = sorted(output_dir.rglob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)

    return {
        "status": "ok",
        "files": [{"name": f.stem, "path": str(f.relative_to(output_dir))} for f in files],
    }
