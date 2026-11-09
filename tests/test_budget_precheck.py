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
    """Assert that budget guard pre-check blocks LLM before spending tokens.

    Specifically verifies mock SDK generate_content was NEVER called.
    """
    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 500
    mock_client.models.count_tokens.return_value = mock_count_resp

    mock_provider = GeminiProvider(
        api_key="mock-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    graph = build_pipeline_graph()

    # Set max_tokens to 100 (< 500 tokens estimated)
    initial_state = create_initial_state(
        source_path=str(sample_docs_dir),
        max_tokens=100,
        output_dir=str(tmp_path / "output"),
        llm_provider=mock_provider,
    )

    result = graph.invoke(initial_state)

    # 1. Critical assertion: SDK generate_content must NEVER be called
    mock_client.models.generate_content.assert_not_called()

    # 2. Check error message
    assert len(result["errors"]) > 0
    assert any(
        "Budget exceeded: estimated 500 input tokens > max 100. LLM call skipped."
        in err
        for err in result["errors"]
    )

    # 3. Frontmatter is None (fallback was used)
    assert result.get("frontmatter") is None
    assert result.get("actual_tokens") == 0

    # 4. Skill is still emitted with fallback frontmatter
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()
    content = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "(placeholder — will be LLM-generated later)" in content


def test_budget_guard_allows_llm_when_within_budget(
    sample_docs_dir: Path, tmp_path: Path
):
    """Assert that budget guard allows LLM call when within budget."""
    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 50
    mock_client.models.count_tokens.return_value = mock_count_resp

    mock_generate_resp = MagicMock()
    mock_generate_resp.text = json.dumps(
        {
            "description": "Synthesized agent skill.",
            "when_to_use": "- Trigger scenario 1\n- Trigger scenario 2",
        }
    )
    mock_generate_resp.usage_metadata.prompt_token_count = 50
    mock_generate_resp.usage_metadata.candidates_token_count = 20
    mock_client.models.generate_content.return_value = mock_generate_resp

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

    # SDK generate_content was called
    mock_client.models.generate_content.assert_called_once()
    assert result["frontmatter"] is not None
    assert result["frontmatter"]["description"] == "Synthesized agent skill."
    assert result["actual_tokens"] == 70
    assert result["llm_input_tokens"] == 50
    assert result["llm_output_tokens"] == 20

    # Skill emitted with synthesized frontmatter
    assert result.get("output_path") is not None
    content = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "description: Synthesized agent skill." in content
    assert "## When to use\n- Trigger scenario 1\n- Trigger scenario 2" in content
