import json
from pathlib import Path

SETTINGS_FILE = Path.home() / ".footage-studio" / "settings.json"


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text())
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def get_left_camera_dir() -> Path | None:
    d = load_settings().get("left_camera_dir")
    return Path(d) if d else None


def get_right_camera_dir() -> Path | None:
    d = load_settings().get("right_camera_dir")
    return Path(d) if d else None


def get_output_dir() -> Path | None:
    d = load_settings().get("output_dir")
    return Path(d) if d else None
