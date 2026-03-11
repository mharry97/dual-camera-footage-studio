import static_ffmpeg

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from footage_studio.web.routers import pages, settings

static_ffmpeg.add_paths()

app = FastAPI(title="Footage Studio")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")

app.include_router(pages.router)
app.include_router(settings.router)


def main():
    import uvicorn
    uvicorn.run("footage_studio.main:app", host="127.0.0.1", port=8000, reload=True)
