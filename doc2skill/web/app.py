import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from doc2skill.llm.gemini import DEFAULT_MODEL
from doc2skill.web.events import pipeline_event_stream

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


class BuildRequest(BaseModel):
    """Request body schema for POST /api/build."""

    source: str
    source_type: Literal["url", "markdown_dir"] = "url"
    max_pages: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=50_000, ge=1)
    no_cache: bool = False
    no_llm: bool = False


def get_llm_config() -> dict[str, str | bool]:
    """Read Gemini runtime config from environment variables without loading .env."""
    model = os.environ.get("DOC2SKILL_LLM_MODEL") or DEFAULT_MODEL
    has_api_key = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "model": model,
        "has_api_key": has_api_key,
    }


def _validate_build_request(payload: BuildRequest) -> str:
    """Validate semantic source requirements; return normalized source string."""
    source = payload.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Field 'source' must not be empty.")

    if payload.source_type == "url":
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL source: {source!r}. Expected http:// or https:// URL.",
            )
    elif payload.source_type == "markdown_dir":
        local_dir = Path(source)
        if not local_dir.exists() or not local_dir.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {source!r}.",
            )

    return source


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

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert Pydantic/FastAPI validation errors (422) into HTTP 400 JSON."""
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"detail": exc.errors()}),
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the single-page web UI."""
        return templates.TemplateResponse(request, "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        """Return health status for readiness and --check verification."""
        return {"status": "ok"}

    @app.post("/api/build")
    async def build(
        payload: BuildRequest, request: Request
    ) -> EventSourceResponse:
        """Stream skill build pipeline execution over Server-Sent Events."""
        source = _validate_build_request(payload)

        return EventSourceResponse(
            pipeline_event_stream(
                source=source,
                source_type=payload.source_type,
                max_tokens=payload.max_tokens,
                no_llm=payload.no_llm,
                request=request,
            ),
            sep="\n",
        )

    return app
