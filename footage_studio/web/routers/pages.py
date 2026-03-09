from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/group", response_class=HTMLResponse)
async def group(request: Request):
    return templates.TemplateResponse("group.html", {"request": request})


@router.get("/stitch", response_class=HTMLResponse)
async def stitch(request: Request):
    return templates.TemplateResponse("stitch.html", {"request": request})


@router.get("/browse", response_class=HTMLResponse)
async def browse(request: Request):
    return templates.TemplateResponse("browse.html", {"request": request})
