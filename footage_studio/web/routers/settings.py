from fastapi import APIRouter
from pydantic import BaseModel

from footage_studio.core import load_settings, save_settings

router = APIRouter(prefix="/api/settings")


class Settings(BaseModel):
    footage_dir: str


@router.get("")
async def get_settings():
    return load_settings()


@router.post("")
async def post_settings(settings: Settings):
    data = load_settings()
    data["footage_dir"] = settings.footage_dir
    save_settings(data)
    return {"ok": True}
