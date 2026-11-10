import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
