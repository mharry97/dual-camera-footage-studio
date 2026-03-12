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


def get_footage_dir() -> Path | None:
    footage_dir = load_settings().get("footage_dir")
    return Path(footage_dir) if footage_dir else None
