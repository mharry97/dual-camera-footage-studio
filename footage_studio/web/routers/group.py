from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footage_studio.core import check_subdirectories, get_footage_dir, set_metadata
from footage_studio.processing import concatenate, scan_camera_dir

router = APIRouter(prefix="/api/group")

CAMERA_SIDES = {
    "Left Camera": "LEFT",
    "Right Camera": "RIGHT",
}


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

    left_groups = scan_camera_dir(footage_dir / "Left Camera")
    right_groups = scan_camera_dir(footage_dir / "Right Camera")

    def serialise_groups(groups):
        return [
            {
                "name": g.name,
                "output_name": g.output_name,
                "total_duration": g.total_duration,
                "files": [{"name": f.name} for f in g.files],
            }
            for g in groups
        ]

    return {
        "status": "ok",
        "left": serialise_groups(left_groups),
        "right": serialise_groups(right_groups),
    }


@router.post("/create-dirs")
async def create_dirs():
    footage_dir = get_footage_dir()
    if not footage_dir:
        return JSONResponse({"status": "no_directory"})

    for name in ["Left Camera", "Right Camera", "Output Footage"]:
        (footage_dir / name).mkdir(parents=True, exist_ok=True)

    return {"status": "ok"}


@router.post("/confirm")
async def confirm():
    footage_dir = get_footage_dir()
    if not footage_dir:
        return JSONResponse({"status": "no_directory"}, status_code=400)

    for dir_name, camera_side in CAMERA_SIDES.items():
        camera_dir = footage_dir / dir_name
        groups = scan_camera_dir(camera_dir)

        for group in groups:
            output_path = camera_dir / group.output_name
            concatenate(
                filepaths=[f.path for f in group.files],
                output_path=output_path,
                metadata={"status": "GROUPED", "camera_side": camera_side},
            )
            for fi in group.files:
                set_metadata(fi.path, "status", "PROCESSED")

    return {"status": "ok"}
