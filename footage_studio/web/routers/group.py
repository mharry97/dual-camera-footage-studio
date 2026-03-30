from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footage_studio.core import (
    get_created_time,
    get_left_camera_dir,
    get_right_camera_dir,
)
from footage_studio.processing import concatenate, scan_camera_dir

router = APIRouter(prefix="/api/group")


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

    left_groups = [g for g in scan_camera_dir(left_dir) if len(g.files) > 1]
    right_groups = [g for g in scan_camera_dir(right_dir) if len(g.files) > 1]

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


@router.post("/confirm")
async def confirm():
    left_dir = get_left_camera_dir()
    right_dir = get_right_camera_dir()
    if not left_dir or not right_dir:
        return JSONResponse({"status": "no_directory"}, status_code=400)

    from wakepy import keep

    with keep.running(on_fail="pass"):
        for camera_dir in [left_dir, right_dir]:
            groups = [g for g in scan_camera_dir(camera_dir) if len(g.files) > 1]

            for group in groups:
                output_path = camera_dir / group.output_name
                first_created = get_created_time(group.files[0].path)
                concatenate(
                    filepaths=[f.path for f in group.files],
                    output_path=output_path,
                    metadata={
                        "creation_time": first_created.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    },
                )
                processed_dir = camera_dir / "_processed"
                processed_dir.mkdir(exist_ok=True)
                for fi in group.files:
                    fi.path.rename(processed_dir / fi.path.name)

    return {"status": "ok"}
