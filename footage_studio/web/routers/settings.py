from fastapi import APIRouter
from pydantic import BaseModel
import json
from pathlib import Path

router = APIRouter(prefix="/api/settings")

SETTINGS_FILE = Path.home() / ".footage-studio" / "settings.json"


def _load() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text())
    return {}


def _save(data: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


class Settings(BaseModel):
    footage_dir: str


@router.get("")
async def get_settings():
    return _load()


@router.post("")
async def post_settings(settings: Settings):
    data = _load()
    data["footage_dir"] = settings.footage_dir
    _save(data)
    return {"ok": True}
