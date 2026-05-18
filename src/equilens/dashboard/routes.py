"""Jinja2 HTML page routes for the EquiLens dashboard."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/audit", response_class=HTMLResponse)
async def page_audit(request: Request):
    return templates.TemplateResponse("audit.html", {"request": request})


@router.get("/generate", response_class=HTMLResponse)
async def page_generate(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})


@router.get("/analyze", response_class=HTMLResponse)
async def page_analyze(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})


@router.get("/jobs", response_class=HTMLResponse)
async def page_jobs(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})


@router.get("/results", response_class=HTMLResponse)
async def page_results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})
