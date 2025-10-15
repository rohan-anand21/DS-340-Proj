from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from ..dependencies import templates

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def get_home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/architecture", response_class=HTMLResponse)
async def get_architecture_page(request: Request):
    return templates.TemplateResponse("architecture.html", {"request": request})

@router.get("/visuals", response_class=HTMLResponse)
async def get_visuals_page(request: Request):
    return templates.TemplateResponse("visuals.html", {"request": request})