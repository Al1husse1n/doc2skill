import os
import re
from pathlib import Path

import pytest

from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS") or not os.getenv("GEMINI_API_KEY"),
    reason="Live LLM test (set RUN_LIVE_TESTS=1 and GEMINI_API_KEY)",
)
def test_gemini_frontmatter_eval(tmp_path: Path):
    """End-to-end evaluation using real Gemini API on stripe_like documentation fixture."""
    fixture_dir = Path(__file__).parent / "fixtures" / "stripe_like"
    assert fixture_dir.exists(), f"Fixture directory missing: {fixture_dir}"

    output_dir = tmp_path / "eval_output"
    graph = build_pipeline_graph()

    initial_state = create_initial_state(
        source_path=str(fixture_dir),
        max_tokens=50_000,
        output_dir=str(output_dir),
        no_llm=False,
    )

    result = graph.invoke(initial_state)

    # 0. Pipeline must succeed without errors
    assert not result.get("errors"), f"Pipeline errors: {result.get('errors')}"
    assert result.get("output_path"), "Output path was not populated"

    skill_file = Path(result["output_path"])
    assert skill_file.exists(), f"Generated skill file does not exist: {skill_file}"
    content = skill_file.read_text(encoding="utf-8")

    # 1. Contains a ## When to use section with at least 2 bullet points
    assert "## When to use" in content
    when_to_use_match = re.search(
        r"## When to use\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    assert when_to_use_match is not None, "Could not extract 'When to use' section"
    when_to_use_text = when_to_use_match.group(1).strip()
    bullet_points = [
        line
        for line in when_to_use_text.splitlines()
        if line.strip().startswith("-") or line.strip().startswith("*")
    ]
    assert len(bullet_points) >= 2, (
        f"Expected at least 2 bullet points in 'When to use', got: {bullet_points}"
    )

    # 2. Contains a mention of at least one of charges, customers, webhooks in frontmatter description
    desc_match = re.search(r"description:\s*(.+)", content)
    assert desc_match is not None, "Frontmatter description not found"
    description = desc_match.group(1).lower()
    has_keyword = any(
        kw in description
        for kw in ("charges", "customers", "webhooks", "payments", "payment")
    )
    assert has_keyword, (
        f"Expected description to mention charges, customers, or webhooks, got: '{description}'"
    )

    # 3. Does NOT contain the string 'placeholder — will be LLM-generated'
    assert "placeholder — will be LLM-generated" not in content

    # 4. Total output tokens is less than 10,000
    actual_tokens = result.get("actual_tokens", 0)
    assert actual_tokens < 10_000, f"Token spend exceeded sanity bound: {actual_tokens}"
