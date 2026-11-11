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
    max_output_tokens: int = Field(default=15_000, ge=1)
    max_context_tokens: int = Field(default=500_000, ge=1)
    no_cache: bool = False
    no_llm: bool = False
    no_condense: bool = False


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


import os as _os
import secrets
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Simple per-IP sliding-window rate limiter scoped to an app instance."""

    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record an attempt for key; return (allowed, retry_after_seconds)."""
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        bucket = self._hits[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = max(1, int(bucket[0] + self.window_seconds - current) + 1)
            return False, retry_after
        bucket.append(current)
        return True, 0


def create_app() -> FastAPI:
    """Create and configure the local doc2skill FastAPI application."""
    app = FastAPI(
        title="doc2skill",
        description="Local web interface for converting documentation into agent skills.",
        docs_url=None,
        redoc_url=None,
    )

    rate_limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60.0)
    app.state.rate_limiter = rate_limiter

    def _client_ip(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip() or "unknown"
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    @app.middleware("http")
    async def auth_and_rate_limit_middleware(request: Request, call_next):
        """Enforce optional DOC2SKILL_API_KEY and per-IP rate limiting on POST /api/build."""
        if request.method == "POST" and request.url.path == "/api/build":
            expected_key = _os.getenv("DOC2SKILL_API_KEY", "").strip()
            if expected_key:
                provided_key = request.headers.get("x-api-key", "").strip()
                if not provided_key or not secrets.compare_digest(provided_key, expected_key):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key."},
                    )

            allowed, retry_after = rate_limiter.check(_client_ip(request))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={"detail": "Rate limit exceeded (max 5 requests per minute)."},
                )

        return await call_next(request)

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
                max_pages=payload.max_pages,
                max_tokens=payload.max_tokens,
                max_output_tokens=payload.max_output_tokens,
                max_context_tokens=payload.max_context_tokens,
                no_cache=payload.no_cache,
                no_llm=payload.no_llm,
                no_condense=payload.no_condense,
                request=request,
            ),
            sep="\n",
        )

    return app
