import json
from pathlib import Path

import httpx
import pytest

from doc2skill.distill.condense import CondensedResult
from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult
from doc2skill.llm.gemini import GeminiProvider
from doc2skill.web.app import create_app


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate disk cache per test so ~/.cache/doc2skill is never read or polluted."""
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path / "web_cache"))


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

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        return CondensedResult(
            body="## API surface\n`SDK.init()`\n\n## Common workflows\n1. Run init.\n\n## Gotchas\nNone.\n\n## Reference\nquickstart",
            sources=[d.path for d in documents],
            input_tokens=200,
            output_tokens=100,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

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
    assert complete_payload["cache_hit"] is False
    assert complete_payload["truncated"] is False
    assert "dropped_documents" not in complete_payload
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

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        raise AssertionError("condense must not be called over budget")

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

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


def test_app_factory_routes():
    """Smoke test importing app factory from doc2skill.web and inspecting routes."""
    from doc2skill.web import create_app as exported_create_app

    app = exported_create_app()
    route_paths = {getattr(r, "path", None) for r in app.routes}
    assert "/" in route_paths
    assert "/api/health" in route_paths
    assert "/api/build" in route_paths


def test_serve_command_smoke(monkeypatch: pytest.MonkeyPatch):
    """Smoke test `doc2skill serve` CLI options, banner, warning, and 0.0.0.0 prompt."""
    from typer.testing import CliRunner

    from doc2skill.cli import app as cli_app

    runner = CliRunner()
    calls: list[dict] = []

    def fake_uvicorn_run(target, **kwargs):
        calls.append({"target": target, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 1. Default invocation: 127.0.0.1:8000, missing GEMINI_API_KEY warns but still runs
    res_default = runner.invoke(cli_app, ["serve"])
    assert res_default.exit_code == 0
    assert "Warning: GEMINI_API_KEY is not set" in res_default.output
    assert "doc2skill server starting" in res_default.output
    assert "http://127.0.0.1:8000" in res_default.output
    assert calls[-1]["host"] == "127.0.0.1"
    assert calls[-1]["port"] == 8000

    # 2. --host 0.0.0.0 without --yes prompts and aborts on 'n'
    res_abort = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"], input="n\n")
    assert res_abort.exit_code != 0
    assert "expose the app to your network" in res_abort.output

    # 3. --host 0.0.0.0 --yes --reload runs factory with reload=True
    res_reload = runner.invoke(
        cli_app, ["serve", "--host", "0.0.0.0", "--port", "8765", "--reload", "--yes"]
    )
    assert res_reload.exit_code == 0
    assert calls[-1]["target"] == "doc2skill.web.app:create_app"
    assert calls[-1]["factory"] is True
    assert calls[-1]["reload"] is True
    assert calls[-1]["host"] == "0.0.0.0"
    assert calls[-1]["port"] == 8765


@pytest.mark.asyncio
async def test_build_respects_max_pages(monkeypatch: pytest.MonkeyPatch):
    """Verify BuildRequest.max_pages passes through pipeline to load_url."""
    captured: dict[str, int] = {}

    async def fake_load_url(url: str, max_pages: int = 100):
        captured["max_pages"] = max_pages
        return [Document(path=url, title="x", content="y")]

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://fake-maxpages.example.com/",
                "source_type": "url",
                "max_pages": 7,
                "max_tokens": 50_000,
                "no_cache": True,
                "no_llm": True,
            },
        )

    assert resp.status_code == 200
    assert captured["max_pages"] == 7


@pytest.mark.asyncio
async def test_build_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_docs: list[Document]
):
    """Verify web builds use disk cache cleanly and honor no_cache=true (skip read & write)."""
    cache_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")

    async def fake_load_url(url: str, max_pages: int = 100):
        return fake_docs

    llm_calls = 0

    def fake_count_tokens(self, prompt: str) -> int:
        return 100

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        nonlocal llm_calls
        llm_calls += 1
        return FrontmatterResult(
            description=f"Synthesized skill run {llm_calls}.",
            when_to_use="- When testing cache hit",
            input_tokens=90,
            output_tokens=20,
        )

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        return CondensedResult(
            body=f"## API surface\n`run_{llm_calls}()`\n\n## Common workflows\n1. Call.\n\n## Gotchas\nNone.\n\n## Reference\ndocs",
            sources=[d.path for d in documents],
            input_tokens=150,
            output_tokens=80,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. First POST with no_cache=false -> cache miss, LLM called once
        r1 = await client.post(
            "/api/build",
            json={
                "source": "https://cached-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": False,
            },
        )
        assert r1.status_code == 200
        p1 = dict(_parse_sse(r1.text))
        assert llm_calls == 1
        assert p1["complete"]["cache_hit"] is False
        assert "Synthesized skill run 1." in p1["complete"]["skill_md"]

        # 2. Second POST with no_cache=false -> cache hit, LLM NOT called
        r2 = await client.post(
            "/api/build",
            json={
                "source": "https://cached-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": False,
            },
        )
        assert r2.status_code == 200
        p2 = dict(_parse_sse(r2.text))
        assert llm_calls == 1
        assert p2["complete"]["cache_hit"] is True
        assert "Synthesized skill run 1." in p2["complete"]["skill_md"]

        # 3. Third POST with no_cache=true -> skips read AND write, LLM called again
        r3 = await client.post(
            "/api/build",
            json={
                "source": "https://cached-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": True,
            },
        )
        assert r3.status_code == 200
        p3 = dict(_parse_sse(r3.text))
        assert llm_calls == 2
        assert p3["complete"]["cache_hit"] is False
        assert "Synthesized skill run 2." in p3["complete"]["skill_md"]

        # 4. Verify third POST skipped writing by posting no_cache=false -> still returns run 1
        r4 = await client.post(
            "/api/build",
            json={
                "source": "https://cached-docs.example.com/",
                "source_type": "url",
                "max_pages": 50,
                "max_tokens": 50_000,
                "no_cache": False,
            },
        )
        assert r4.status_code == 200
        p4 = dict(_parse_sse(r4.text))
        assert llm_calls == 2
        assert p4["complete"]["cache_hit"] is True
        assert "Synthesized skill run 1." in p4["complete"]["skill_md"]


@pytest.mark.asyncio
async def test_build_sse_truncated(monkeypatch: pytest.MonkeyPatch):
    """Mock 100 docs exceeding context limit -> assert truncate_warning and complete.dropped_documents."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")

    # 100 fake docs (~10k tokens each -> ~1M tokens total > 500k context limit)
    huge_docs = [
        Document(
            path=f"https://docs.example.com/deep/page_{i:02d}"
            if i >= 50
            else f"https://docs.example.com/page_{i:02d}",
            content="Z" * 40_000,
            title=f"Page {i}",
        )
        for i in range(100)
    ]

    captured: dict[str, int] = {}

    async def fake_load_url(url: str, max_pages: int = 100):
        return huge_docs

    def fake_count_tokens(self, prompt: str) -> int:
        if "Documentation snippets" in prompt:
            return 100
        return prompt.count("=== DOCUMENT:") * 2000

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        return FrontmatterResult(
            description="Truncated docs skill.",
            when_to_use="- When working with truncated corpus",
            input_tokens=100,
            output_tokens=20,
        )

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        captured["n_docs"] = len(documents)
        return CondensedResult(
            body=(
                "## API surface\n`Client()`\n\n"
                "## Common workflows\n1. Use client.\n\n"
                "## Gotchas\nTruncation applied.\n\n"
                "## Reference\ncore"
            ),
            sources=[d.path for d in documents],
            input_tokens=len(documents) * 2000,
            output_tokens=1_500,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://docs.example.com/",
                "source_type": "url",
                "max_pages": 100,
                "max_tokens": 1_000_000,
                "max_context_tokens": 500_000,
                "no_cache": True,
            },
        )

    assert resp.status_code == 200
    raw_sse_bytes = resp.content
    print(f"\n[test_build_sse_truncated] n_docs={captured.get('n_docs')}")
    print(raw_sse_bytes.decode("utf-8"))

    assert captured["n_docs"] == 49
    assert raw_sse_bytes.index(b"event: budget_pre") > raw_sse_bytes.index(
        b"event: truncate_warning"
    )

    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    assert "truncate_warning" in names
    assert "condensed" in names
    assert names.index("ingest_done") < names.index("truncate_warning") < names.index("budget_pre")

    events_map = dict(parsed)
    assert events_map["truncate_warning"]["reason"] == "context_limit"
    assert events_map["truncate_warning"]["dropped_count"] == 51
    assert events_map["budget_pre"]["estimated_input_tokens"] == 49 * 2000 + 100
    assert events_map["complete"]["truncated"] is True
    assert len(events_map["complete"]["dropped_documents"]) == 51


@pytest.mark.asyncio
async def test_build_sse_condensed(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """Mock condense and assert condensed SSE event includes token counts and sections."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")

    async def fake_load_url(url: str, max_pages: int = 100):
        return fake_docs

    def fake_count_tokens(self, prompt: str) -> int:
        return 100

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        return FrontmatterResult(
            description="Condensed SSE event skill.",
            when_to_use="- When checking condensed SSE event",
            input_tokens=80,
            output_tokens=20,
        )

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        return CondensedResult(
            body=(
                "## API surface\n`Client.get()`\n\n"
                "## Common workflows\n1. Send GET.\n\n"
                "## Gotchas\nTimeouts.\n\n"
                "## Reference\nclient"
            ),
            sources=[d.path for d in documents],
            input_tokens=8400,
            output_tokens=4200,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

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
                "no_cache": True,
            },
        )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    assert names.index("llm_done") < names.index("condensed") < names.index("complete")

    condensed_payload = dict(parsed)["condensed"]
    assert condensed_payload["input_tokens"] == 8400
    assert condensed_payload["output_tokens"] == 4200
    assert condensed_payload["sections"] == [
        "API surface",
        "Common workflows",
        "Gotchas",
        "Reference",
    ]


@pytest.mark.asyncio
async def test_build_sse_validation(monkeypatch: pytest.MonkeyPatch):
    """Mock condense with 1 verified + 1 unverified identifier -> assert validation SSE and complete dict."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")

    docs = [
        Document(
            path="https://fake-docs.example.com/client",
            content="# Client\nUse `httpx.Client` for HTTP requests.",
            title="Client",
        )
    ]

    async def fake_load_url(url: str, max_pages: int = 100):
        return docs

    def fake_count_tokens(self, prompt: str) -> int:
        return 100

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        return FrontmatterResult(
            description="Validation SSE test skill.",
            when_to_use="- When checking validation SSE output",
            input_tokens=80,
            output_tokens=20,
        )

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        return CondensedResult(
            body=(
                "## API surface\n`httpx.Client(timeout=DEFAULT_TIMEOUT)` and `httpx.FictionalClient`\n\n"
                "## Common workflows\n1. Use `httpx.Client`.\n\n"
                "## Gotchas\nAvoid `httpx.FictionalClient`.\n\n"
                "## Reference\nclient"
            ),
            sources=[d.path for d in documents],
            input_tokens=300,
            output_tokens=120,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

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
                "no_cache": True,
            },
        )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    names = [name for name, _ in parsed]
    assert names.index("llm_done") < names.index("validation") < names.index("complete")

    events_map = dict(parsed)
    val_event = events_map["validation"]
    assert val_event["verified_count"] == 1
    assert val_event["known_external_count"] == 0
    assert val_event["source_divergent_count"] == 1
    assert val_event["unverified_count"] == 1
    assert val_event["precision"] == 0.5
    assert val_event["inference_rate"] == 0.333
    assert val_event["source_divergent_preview"] == [
        {"text": "DEFAULT_TIMEOUT", "kind": "env_var"}
    ]
    assert val_event["unverified_preview"] == ["httpx.FictionalClient"]

    complete_val = events_map["complete"]["validation"]
    assert complete_val["verified_count"] == 1
    assert complete_val["known_external_count"] == 0
    assert complete_val["source_divergent_count"] == 1
    assert complete_val["unverified_count"] == 1
    assert complete_val["precision"] == 0.5
    assert complete_val["inference_rate"] == 0.333
    assert len(complete_val["source_divergent"]) == 1
    assert complete_val["source_divergent"][0]["text"] == "DEFAULT_TIMEOUT"
    assert len(complete_val["unverified"]) == 1
    assert complete_val["unverified"][0]["text"] == "httpx.FictionalClient"
    assert complete_val["unverified"][0]["kind"] == "class"
    assert "httpx.FictionalClient" in complete_val["unverified"][0]["context"]


@pytest.mark.asyncio
async def test_build_supabase_cache_hit_before_load_url(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """Supabase cache is checked before load_url; hits skip crawling and emit cache_hit."""
    from doc2skill.cache import supabase as supabase_cache

    call_order: list[str] = []
    set_calls: list[tuple[str, int, str, str]] = []
    cached_md = "---\nname: cached-skill\ndescription: cached\n---\n\n# Cached Skill\n"

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        call_order.append("load_url")
        return fake_docs

    def fake_get_hit(source: str, max_tokens: int, model: str) -> str | None:
        call_order.append("get_cached_skill")
        return cached_md

    def fake_set(source: str, max_tokens: int, model: str, skill_md: str) -> None:
        call_order.append("set_cached_skill")
        set_calls.append((source, max_tokens, model, skill_md))

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)
    monkeypatch.setattr(supabase_cache, "get_cached_skill", fake_get_hit)
    monkeypatch.setattr(supabase_cache, "set_cached_skill", fake_set)

    # 1. Cache hit -> load_url must NOT be called; emits cache_hit and complete(cache_hit=True)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://cached.example.com/docs",
                "source_type": "url",
                "max_pages": 10,
                "max_tokens": 50_000,
                "no_cache": False,
                "no_llm": True,
            },
        )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    event_names = [name for name, _ in parsed]
    assert "cache_hit" in event_names
    assert event_names[-1] == "complete"
    data_by_event = dict(parsed)
    assert data_by_event["cache_hit"]["skill_md"] == cached_md
    assert data_by_event["complete"]["cache_hit"] is True
    assert data_by_event["complete"]["skill_md"] == cached_md
    assert call_order == ["get_cached_skill"]
    assert "load_url" not in call_order

    # 2. Cache miss -> get_cached_skill MUST precede load_url
    call_order.clear()
    monkeypatch.setattr(supabase_cache, "get_cached_skill", lambda *a, **kw: (call_order.append("get_cached_skill"), None)[1])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_miss = await client.post(
            "/api/build",
            json={
                "source": "https://miss.example.com/docs",
                "source_type": "url",
                "max_pages": 10,
                "max_tokens": 50_000,
                "no_cache": False,
                "no_llm": True,
            },
        )
    assert resp_miss.status_code == 200
    assert call_order.index("get_cached_skill") < call_order.index("load_url")
    assert len(set_calls) == 1

    # 3. no_cache=True -> neither get_cached_skill nor set_cached_skill is called
    call_order.clear()
    set_calls.clear()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_bypass = await client.post(
            "/api/build",
            json={
                "source": "https://nocache.example.com/docs",
                "source_type": "url",
                "max_pages": 10,
                "max_tokens": 50_000,
                "no_cache": True,
                "no_llm": True,
            },
        )
    assert resp_bypass.status_code == 200
    assert "get_cached_skill" not in call_order
    assert "set_cached_skill" not in call_order
    assert "load_url" in call_order


@pytest.mark.asyncio
async def test_build_auth_required_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """POST /api/build enforces X-API-Key when DOC2SKILL_API_KEY is set; read routes stay open."""

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)

    payload = {
        "source": "https://auth-test.example.com/docs",
        "source_type": "url",
        "max_pages": 5,
        "max_tokens": 50_000,
        "no_cache": True,
        "no_llm": True,
    }

    # 1. Local mode (unset DOC2SKILL_API_KEY) allows unauthenticated /api/build
    monkeypatch.delenv("DOC2SKILL_API_KEY", raising=False)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_open = await client.post("/api/build", json=payload)
        assert resp_open.status_code == 200

    # 2. Protected mode requires exact X-API-Key on POST /api/build
    monkeypatch.setenv("DOC2SKILL_API_KEY", "super-secret-build-key")
    protected_app = create_app()
    protected_transport = httpx.ASGITransport(app=protected_app)
    async with httpx.AsyncClient(transport=protected_transport, base_url="http://test") as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/api/health")).status_code == 200

        resp_missing = await client.post("/api/build", json=payload)
        assert resp_missing.status_code == 401

        resp_wrong = await client.post(
            "/api/build",
            headers={"X-API-Key": "wrong-key"},
            json=payload,
        )
        assert resp_wrong.status_code == 401

        resp_valid = await client.post(
            "/api/build",
            headers={"X-API-Key": "super-secret-build-key"},
            json=payload,
        )
        assert resp_valid.status_code == 200
        assert _parse_sse(resp_valid.text)[-1][0] == "complete"


@pytest.mark.asyncio
async def test_build_rate_limit_429_on_sixth_request(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """POST /api/build limits each IP to 5 requests per minute and returns 429 on the 6th."""
    monkeypatch.delenv("DOC2SKILL_API_KEY", raising=False)

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)

    payload = {
        "source": "https://ratelimit.example.com/docs",
        "source_type": "url",
        "max_pages": 5,
        "max_tokens": 50_000,
        "no_cache": True,
        "no_llm": True,
    }

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(5):
            resp = await client.post("/api/build", json=payload)
            assert resp.status_code == 200, f"Request {i + 1} unexpectedly returned {resp.status_code}"

        resp_sixth = await client.post("/api/build", json=payload)
        assert resp_sixth.status_code == 429
        assert "retry-after" in resp_sixth.headers
        assert "Rate limit exceeded" in resp_sixth.json()["detail"]

        # Separate client IP still has its own 5 req/min allowance
        resp_other_ip = await client.post(
            "/api/build",
            headers={"X-Forwarded-For": "203.0.113.42"},
            json=payload,
        )
        assert resp_other_ip.status_code == 200


@pytest.mark.asyncio
async def test_build_rate_limit_by_api_key_ignores_ip_spoofing(
    monkeypatch: pytest.MonkeyPatch, fake_docs: list[Document]
):
    """When X-API-Key is present, rate limits key on hashed API key so X-Forwarded-For cannot bypass."""
    monkeypatch.setenv("DOC2SKILL_API_KEY", "prod-secret-token")

    async def fake_load_url(url: str, max_pages: int = 100, follow_sitemap: bool = True):
        return fake_docs

    monkeypatch.setattr("doc2skill.ingest.url.load_url", fake_load_url)

    payload = {
        "source": "https://ratelimit-key.example.com/docs",
        "source_type": "url",
        "max_pages": 5,
        "max_tokens": 50_000,
        "no_cache": True,
        "no_llm": True,
    }

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(5):
            resp = await client.post(
                "/api/build",
                headers={
                    "X-API-Key": "prod-secret-token",
                    "X-Forwarded-For": f"198.51.100.{i + 1}",
                },
                json=payload,
            )
            assert resp.status_code == 200

        # 6th call with yet another spoofed IP still hits the API-key bucket -> 429
        resp_sixth = await client.post(
            "/api/build",
            headers={
                "X-API-Key": "prod-secret-token",
                "X-Forwarded-For": "198.51.100.99",
            },
            json=payload,
        )
        assert resp_sixth.status_code == 429

        # Bucket keys stored in memory are hashed ("key:<16 hex>"), never raw keys
        stored_keys = list(app.state.rate_limiter._hits.keys())
        assert len(stored_keys) == 1
        assert stored_keys[0].startswith("key:")
        assert len(stored_keys[0]) == 4 + 16
        assert "prod-secret-token" not in stored_keys[0]







