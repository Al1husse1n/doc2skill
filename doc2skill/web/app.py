from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the local doc2skill FastAPI application."""
    app = FastAPI(
        title="doc2skill",
        description="Local web interface for converting documentation into agent skills.",
        docs_url=None,
        redoc_url=None,
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the single-page web UI."""
        return templates.TemplateResponse(request, "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        """Return health status for readiness and --check verification."""
        return {"status": "ok"}

    return app
