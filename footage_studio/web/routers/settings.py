from fastapi import APIRouter
from pydantic import BaseModel

from footage_studio.core import load_settings, save_settings

router = APIRouter(prefix="/api/settings")


class Settings(BaseModel):
    left_camera_dir: str
    right_camera_dir: str
    output_dir: str


@router.get("")
async def get_settings():
    return load_settings()


@router.post("")
async def post_settings(settings: Settings):
    data = load_settings()
    data["left_camera_dir"] = settings.left_camera_dir
    data["right_camera_dir"] = settings.right_camera_dir
    data["output_dir"] = settings.output_dir
    save_settings(data)
    return {"ok": True}
