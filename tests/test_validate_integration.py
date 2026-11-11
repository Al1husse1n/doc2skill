import json
from pathlib import Path
from unittest.mock import MagicMock

from doc2skill.ingest.base import Document
from doc2skill.llm.gemini import GeminiProvider
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


def _mock_provider(condensed_body: str) -> GeminiProvider:
    """Create a mocked GeminiProvider returning deterministic frontmatter and condensed body."""
    mock_client = MagicMock()
    count_resp = MagicMock()
    count_resp.total_tokens = 100
    mock_client.models.count_tokens.return_value = count_resp

    fm_resp = MagicMock()
    fm_resp.text = json.dumps(
        {
            "description": "HTTP client skill.",
            "when_to_use": "- Making HTTP requests",
        }
    )
    fm_resp.usage_metadata.prompt_token_count = 50
    fm_resp.usage_metadata.candidates_token_count = 20

    condense_resp = MagicMock()
    condense_resp.text = condensed_body
    condense_resp.usage_metadata.prompt_token_count = 150
    condense_resp.usage_metadata.candidates_token_count = 120

    mock_client.models.generate_content.side_effect = [fm_resp, condense_resp]
    return GeminiProvider(api_key="mock-key", model="gemini-2.5-flash", client=mock_client)


def test_pipeline_validate_reports_unverified_and_ships_by_default(tmp_path: Path):
    """Default mode emits SKILL.md and attaches validation_report with 1 unverified identifier."""
    docs = [
        Document(
            path="client.md",
            content="# Client\nUse `httpx.Client` for pooled synchronous requests.",
            title="Client",
        )
    ]
    body = (
        "## API surface\n`httpx.Client` and `httpx.FictionalClient`\n\n"
        "## Common workflows\n1. Instantiate `httpx.Client()`.\n\n"
        "## Gotchas\nAvoid `httpx.FictionalClient` if missing.\n\n"
        "## Reference\nclient.md"
    )

    graph = build_pipeline_graph()
    state = create_initial_state(
        source_path="docs",
        documents=docs,
        output_dir=str(tmp_path / "out"),
        llm_provider=_mock_provider(body),
        strict=False,
    )
    result = graph.invoke(state)

    assert not result.get("errors")
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()

    report = result.get("validation_report")
    assert report is not None
    assert [i["text"] for i in report["verified"]] == ["httpx.Client"]
    assert [i["text"] for i in report["unverified"]] == ["httpx.FictionalClient"]
    assert report["precision"] == 0.5


def test_pipeline_validate_strict_fails_and_skips_emit(tmp_path: Path):
    """Strict mode populates errors and skips emit_skill when unverified identifiers exist."""
    docs = [
        Document(
            path="client.md",
            content="# Client\nUse `httpx.Client` for requests.",
            title="Client",
        )
    ]
    body = (
        "## API surface\n`httpx.Client` and `httpx.FictionalClient`\n\n"
        "## Common workflows\n1. Instantiate `httpx.Client()`.\n\n"
        "## Gotchas\nHandle network errors cleanly.\n\n"
        "## Reference\nclient.md"
    )

    output_dir = tmp_path / "strict_out"
    graph = build_pipeline_graph()
    state = create_initial_state(
        source_path="docs",
        documents=docs,
        output_dir=str(output_dir),
        llm_provider=_mock_provider(body),
        strict=True,
    )
    result = graph.invoke(state)

    assert len(result["errors"]) == 1
    assert "Strict mode: 1 unverified identifiers" in result["errors"][0]
    assert "httpx.FictionalClient" in result["errors"][0]
    assert result.get("output_path") is None
    assert result.get("skill_md") is None
    assert not any(output_dir.rglob("SKILL.md")) if output_dir.exists() else True
    assert result["validation_report"] is not None
    assert [i["text"] for i in result["validation_report"]["unverified"]] == [
        "httpx.FictionalClient"
    ]


def test_pipeline_validate_strict_passes_when_all_verified(tmp_path: Path):
    """Strict mode succeeds and emits SKILL.md when every extracted identifier verifies."""
    docs = [
        Document(
            path="client.md",
            content="# Client\nUse `httpx.Client` for synchronous requests.",
            title="Client",
        )
    ]
    body = (
        "## API surface\n`httpx.Client`\n\n"
        "## Common workflows\n1. Use `httpx.Client()` cleanly.\n\n"
        "## Gotchas\nReuse a single `httpx.Client` instance across requests.\n\n"
        "## Reference\nclient.md"
    )

    graph = build_pipeline_graph()
    state = create_initial_state(
        source_path="docs",
        documents=docs,
        output_dir=str(tmp_path / "clean_out"),
        llm_provider=_mock_provider(body),
        strict=True,
    )
    result = graph.invoke(state)

    assert not result.get("errors")
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()
    assert result["validation_report"]["unverified"] == []
    assert result["validation_report"]["precision"] == 1.0


def test_pipeline_validate_strict_allows_source_divergent(tmp_path: Path):
    """Strict mode succeeds and emits SKILL.md when only source_divergent (0 unverified) exists."""
    docs = [
        Document(
            path="client.md",
            content="# Client\nUse `httpx.Client` for synchronous requests.",
            title="Client",
        )
    ]
    body = (
        "## API surface\n`httpx.Client(max_redirects: int = DEFAULT_MAX_REDIRECTS)`\n\n"
        "## Common workflows\n1. Use `httpx.Client(max_redirects=10)`.\n\n"
        "## Gotchas\nRespect redirect defaults.\n\n"
        "## Reference\nclient.md"
    )

    graph = build_pipeline_graph()
    state = create_initial_state(
        source_path="docs",
        documents=docs,
        output_dir=str(tmp_path / "divergent_strict_out"),
        llm_provider=_mock_provider(body),
        strict=True,
    )
    result = graph.invoke(state)

    assert not result.get("errors")
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()
    report = result["validation_report"]
    assert report["unverified"] == []
    assert {i["text"] for i in report["source_divergent"]} == {
        "max_redirects",
        "DEFAULT_MAX_REDIRECTS",
    }
    assert report["precision"] == 1.0
    assert report["inference_rate"] > 0.0

