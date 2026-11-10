import json

import httpx
import pytest

from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult
from doc2skill.llm.gemini import GeminiProvider
from doc2skill.web.app import create_app


def _parse_sse(raw_text: str) -> list[tuple[str, dict]]:
    """Parse SSE stream separated by blank lines (\\n\\n) into (event, payload) pairs."""
    assert "\r\n" not in raw_text, "Expected LF (\\n\\n) SSE block separation, got CRLF."
    blocks = [b for b in raw_text.split("\n\n") if b.strip()]
    events: list[tuple[str, dict]] = []
    for block in blocks:
        event_name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        assert event_name, f"Missing event name in SSE block: {block!r}"
        assert data_lines, f"Missing data payload in SSE block: {block!r}"
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


@pytest.fixture
def fake_docs() -> list[Document]:
    """Two fixture Documents for mocked URL ingestion."""
    return [
        Document(
            path="https://fake-docs.example.com/",
            content="# Fake Docs\nMain documentation entry point for testing SSE streaming.",
            title="Fake Docs",
        ),
        Document(
            path="https://fake-docs.example.com/quickstart",
            content="# Quickstart\nInstall the SDK and configure your client.",
            title="Quickstart",
        ),
    ]


@pytest.mark.asyncio
async def test_health():
    """GET /api/health returns 200 with status ok."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_serves_html():
    """GET / returns 200 HTML containing the form and doc2skill branding."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<form" in resp.text
    assert "doc2skill" in resp.text


@pytest.mark.asyncio
async def test_build_validation():
    """POST /api/build with empty or invalid body returns HTTP 400 JSON (not 200/422, not SSE)."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_empty = await client.post("/api/build", json={})
        assert resp_empty.status_code == 400
        assert "application/json" in resp_empty.headers.get("content-type", "")
        assert "detail" in resp_empty.json()

        resp_bad_url = await client.post(
            "/api/build",
            json={"source": "not-a-valid-url", "source_type": "url"},
        )
        assert resp_bad_url.status_code == 400
        assert "application/json" in resp_bad_url.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_build_sse_success(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """Mock LLM + URL ingest and verify full ordered SSE event sequence."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    def fake_count_tokens(self, prompt: str) -> int:
        return 120

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        return FrontmatterResult(
            description="Synthesized skill from mocked fake docs.",
            when_to_use="- When testing mocked SSE streaming",
            input_tokens=115,
            output_tokens=25,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://fake-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": False,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.text.endswith("\n\n")

    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    expected_sequence = ["start", "ingest_done", "budget_pre", "llm_done", "complete"]

    # Check required events appear in exact relative order
    positions = [names.index(ev) for ev in expected_sequence]
    assert positions == sorted(positions), f"Unexpected order: {names}"

    complete_payload = dict(parsed)["complete"]
    assert complete_payload["skill_md"].startswith("---")
    assert "Synthesized skill from mocked fake docs." in complete_payload["skill_md"]


@pytest.mark.asyncio
async def test_build_sse_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """When max_tokens is exceeded, stream start -> budget_pre -> error with no llm_done."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    def fake_count_tokens(self, prompt: str) -> int:
        return 500

    synth_called = False

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        nonlocal synth_called
        synth_called = True
        raise AssertionError("synthesize_frontmatter must not be called over budget")

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://fake-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 1,
                "no_cache": False,
            },
        )

    assert resp.status_code == 200
    assert not synth_called

    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    assert "llm_done" not in names
    assert "complete" not in names

    expected = ["start", "budget_pre", "error"]
    positions = [names.index(ev) for ev in expected]
    assert positions == sorted(positions), f"Unexpected order: {names}"

    err_payload = dict(parsed)["error"]
    assert err_payload["stage"] == "budget"


@pytest.mark.asyncio
async def test_build_sse_ingest_error(monkeypatch: pytest.MonkeyPatch):
    """When URL ingestion raises httpx.ConnectError, emit SSE error event with stage ingest."""

    async def failing_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        raise httpx.ConnectError("Simulated connection failure")

    monkeypatch.setattr("doc2skill.ingest.url.load_url", failing_load_url)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://unreachable.example.com/",
                "source_type": "url",
                "max_pages": 5,
                "max_tokens": 50_000,
                "no_cache": False,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    parsed = _parse_sse(resp.text)
    assert parsed[-1][0] == "error"
    assert parsed[-1][1]["stage"] == "ingest"
    assert "Simulated connection failure" in parsed[-1][1]["message"]


@pytest.mark.asyncio
async def test_build_no_llm_flag(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """When no_llm is true, skip LLM synthesis and emit SKILL.md with fallback placeholder."""

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("LLM should not be called when no_llm=True")

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fail_if_called)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://fake-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": False,
                "no_llm": True,
            },
        )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    assert "llm_done" not in names
    assert names[-1] == "complete"

    skill_md = dict(parsed)["complete"]["skill_md"]
    assert "(placeholder — will be LLM-generated later)" in skill_md
