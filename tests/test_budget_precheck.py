import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from doc2skill.distill.condense import CondensedResult
from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult
from doc2skill.llm.gemini import GeminiProvider
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary docs directory."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text(
        "# API Reference\nComprehensive API endpoints for agent operations.",
        encoding="utf-8",
    )
    return docs


def test_budget_guard_blocks_llm_when_over_budget(
    sample_docs_dir: Path, tmp_path: Path
):
    """Assert budget guard pre-check blocks both frontmatter & condense before spending tokens."""
    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 300  # 300 frontmatter + 300 condense = 600 > 100
    mock_client.models.count_tokens.return_value = mock_count_resp

    mock_provider = GeminiProvider(
        api_key="mock-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    graph = build_pipeline_graph()

    initial_state = create_initial_state(
        source_path=str(sample_docs_dir),
        max_tokens=100,
        output_dir=str(tmp_path / "output"),
        llm_provider=mock_provider,
    )

    result = graph.invoke(initial_state)

    # 1. Critical assertion: SDK generate_content must NEVER be called for either step
    mock_client.models.generate_content.assert_not_called()

    # 2. Check error message matches sum of both input estimates (300 + 300 = 600)
    assert result["estimated_tokens"] == 600
    assert any(
        "Budget exceeded: estimated 600 input tokens > max 100. LLM call skipped."
        in err
        for err in result["errors"]
    )

    # 3. Fallback frontmatter + fallback raw string body used
    assert result.get("frontmatter") is None
    assert isinstance(result.get("distilled"), str)
    assert "Comprehensive API endpoints" in result["distilled"]
    assert result.get("actual_tokens") == 0

    # 4. Skill is still emitted with fallback frontmatter
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()
    content = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "(placeholder — will be LLM-generated later)" in content


def test_budget_guard_allows_llm_when_within_budget(
    sample_docs_dir: Path, tmp_path: Path
):
    """Assert budget guard allows both frontmatter and condensation calls when within budget."""
    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 50
    mock_client.models.count_tokens.return_value = mock_count_resp

    fm_resp = MagicMock()
    fm_resp.text = json.dumps(
        {
            "description": "Synthesized agent skill.",
            "when_to_use": "- Trigger scenario 1\n- Trigger scenario 2",
        }
    )
    fm_resp.usage_metadata.prompt_token_count = 50
    fm_resp.usage_metadata.candidates_token_count = 20

    condense_resp = MagicMock()
    condense_resp.text = (
        "## API surface\n`Endpoint()`\n\n"
        "## Common workflows\n1. Call API endpoints carefully.\n\n"
        "## Gotchas\nAlways set timeouts.\n\n"
        "## Reference\napi.md"
    )
    condense_resp.usage_metadata.prompt_token_count = 100
    condense_resp.usage_metadata.candidates_token_count = 120

    mock_client.models.generate_content.side_effect = [fm_resp, condense_resp]

    mock_provider = GeminiProvider(
        api_key="mock-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    graph = build_pipeline_graph()

    initial_state = create_initial_state(
        source_path=str(sample_docs_dir),
        max_tokens=1000,
        output_dir=str(tmp_path / "output"),
        llm_provider=mock_provider,
    )

    result = graph.invoke(initial_state)

    # SDK generate_content was called twice: frontmatter + condense
    assert mock_client.models.generate_content.call_count == 2
    assert result["frontmatter"] is not None
    assert result["frontmatter"]["description"] == "Synthesized agent skill."
    assert isinstance(result["distilled"], str)
    assert "## API surface" in result["distilled"]
    assert result["llm_input_tokens"] == 150
    assert result["llm_output_tokens"] == 140
    assert result["actual_tokens"] == 290

    # Skill emitted with synthesized frontmatter and condensed body
    assert result.get("output_path") is not None
    content = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "description: Synthesized agent skill." in content
    assert "## When to use\n- Trigger scenario 1\n- Trigger scenario 2" in content
    assert "## API surface" in content


def test_budget_guard_estimates_post_truncation_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """truncate runs before budget_guard so budget pre-check and condense both see 49 docs."""
    docs = [
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

    def fake_count_tokens(self, prompt: str) -> int:
        if "Documentation snippets" in prompt:
            return 100
        return prompt.count("=== DOCUMENT:") * 2000

    def fake_synthesize(self, documents, max_tokens: int = 1000) -> FrontmatterResult:
        captured["fm_n_docs"] = len(documents)
        return FrontmatterResult(
            description="Post-truncation skill.",
            when_to_use="- Trigger 1\n- Trigger 2",
            input_tokens=100,
            output_tokens=20,
        )

    async def fake_condense(
        documents, model=None, max_output_tokens=15_000, max_context_tokens=500_000
    ):
        captured["condense_n_docs"] = len(documents)
        return CondensedResult(
            body=(
                "## API surface\n`Client()`\n\n"
                "## Common workflows\n1. Use client safely.\n\n"
                "## Gotchas\nRespect context limit.\n\n"
                "## Reference\ncore"
            ),
            sources=[d.path for d in documents],
            input_tokens=len(documents) * 2000,
            output_tokens=400,
            dropped_documents=[],
            truncated=False,
        )

    monkeypatch.setattr(GeminiProvider, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(GeminiProvider, "synthesize_frontmatter", fake_synthesize)
    monkeypatch.setattr("doc2skill.distill.condense.condense", fake_condense)

    provider = GeminiProvider(api_key="mock-key", model="gemini-2.5-flash", client=MagicMock())
    graph = build_pipeline_graph()

    # 100 docs = 200,100 tokens (> 150,000), while 49 post-truncation docs = 98,100 tokens (<= 150,000)
    initial_state = create_initial_state(
        source_path="https://docs.example.com/",
        documents=docs,
        max_tokens=150_000,
        max_context_tokens=500_000,
        output_dir=str(tmp_path / "output"),
        llm_provider=provider,
    )

    result = graph.invoke(initial_state)

    assert not result.get("errors"), f"Unexpected errors: {result.get('errors')}"
    assert result["estimated_tokens"] == 49 * 2000 + 100
    assert result["truncated"] is True
    assert len(result["dropped_documents"]) == 51
    assert captured["fm_n_docs"] == 49
    assert captured["condense_n_docs"] == 49


def test_truncate_skipped_when_no_condense(tmp_path: Path):
    """truncate_node does not drop documents when no_condense=True; budget_guard sees all raw docs."""
    docs = [
        Document(path=f"doc_{i}.md", content="X" * 200, title=f"Doc {i}")
        for i in range(10)
    ]

    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 300
    mock_client.models.count_tokens.return_value = mock_count_resp
    provider = GeminiProvider(api_key="mock-key", model="gemini-2.5-flash", client=mock_client)

    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path="raw_docs",
        documents=docs,
        max_tokens=100,
        max_context_tokens=50,
        output_dir=str(tmp_path / "output"),
        no_condense=True,
        llm_provider=provider,
    )

    result = graph.invoke(initial_state)

    assert result["truncated"] is False
    assert result["dropped_documents"] == []
    assert len(result["documents"]) == 10
    assert any("Budget exceeded" in err for err in result["errors"])
