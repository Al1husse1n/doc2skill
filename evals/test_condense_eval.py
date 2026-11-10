import os
import re
from pathlib import Path

import pytest

from doc2skill.budget.guard import estimate_tokens
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS") or not os.getenv("GEMINI_API_KEY"),
    reason="Set RUN_LIVE_TESTS=1 and GEMINI_API_KEY to run live httpx condensation eval",
)
def test_httpx_docs_condensation_eval(tmp_path: Path):
    """End-to-end live evaluation condensing vendored httpx docs vs uncondensed baseline."""
    fixture_dir = Path(__file__).parent / "fixtures" / "httpx_docs"
    assert fixture_dir.exists(), f"Fixture directory missing: {fixture_dir}"

    graph = build_pipeline_graph()

    # 1. Condensed run (default v5 behavior)
    condensed_out = tmp_path / "condensed_out"
    condensed_state = create_initial_state(
        source_path=str(fixture_dir),
        max_tokens=100_000,
        max_output_tokens=15_000,
        max_context_tokens=500_000,
        output_dir=str(condensed_out),
        no_llm=False,
        no_condense=False,
    )
    result = graph.invoke(condensed_state)

    assert not result.get("errors"), f"Condensed pipeline errors: {result.get('errors')}"
    skill_md = result["skill_md"]
    assert skill_md is not None

    skill_tokens = estimate_tokens(skill_md)
    llm_in = result.get("llm_input_tokens", 0)
    llm_out = result.get("llm_output_tokens", 0)
    actual_tokens = result.get("actual_tokens", 0)
    truncated = bool(result.get("truncated", False))

    # 2. Uncondensed baseline run (--no-condense)
    baseline_out = tmp_path / "baseline_out"
    baseline_state = create_initial_state(
        source_path=str(fixture_dir),
        max_tokens=100_000,
        output_dir=str(baseline_out),
        no_llm=False,
        no_condense=True,
    )
    baseline_result = graph.invoke(baseline_state)
    assert not baseline_result.get("errors"), (
        f"Baseline pipeline errors: {baseline_result.get('errors')}"
    )
    baseline_md = baseline_result["skill_md"]
    assert baseline_md is not None
    baseline_skill_tokens = estimate_tokens(baseline_md)

    print(
        f"\n[EVAL METRICS]"
        f"\nCondensed SKILL.md tokens: {skill_tokens}"
        f"\nLLM output tokens: {llm_out}"
        f"\nLLM input tokens: {llm_in}"
        f"\nTotal actual tokens: {actual_tokens}"
        f"\nTruncated: {truncated}"
    )

    # Assertion 1: Condensed skill fits <= 15,000 token budget, corpus fits context without truncation
    assert llm_out < 15_000, f"LLM output tokens exceeded 15k: {llm_out}"
    assert skill_tokens < 15_000, f"Condensed SKILL.md too large: {skill_tokens} tokens"
    assert result["actual_tokens"] < result["max_context_tokens"], (
        f"Total actual tokens {actual_tokens} exceeded max_context_tokens "
        f"{result['max_context_tokens']}"
    )
    assert not result["truncated"], (
        f"Corpus should fit without truncation, dropped={result.get('dropped_documents')}"
    )

    # Assertion 2: API surface preserved (all required identifiers)
    for ident in ("httpx.Client", "httpx.AsyncClient", "httpx.get", "httpx.post", "httpx.Timeout"):
        assert ident in skill_md, f"Missing required API identifier: {ident}"

    # Assertion 3: Common workflows preserved
    assert any(
        wf in skill_md for wf in ("response.json()", "response.text", "await client.get")
    ), "Missing workflow snippet (response.json(), response.text, or await client.get)"

    # Assertion 4: Gotchas preserved
    lower_md = skill_md.lower()
    assert any(
        gotcha in lower_md for gotcha in ("timeout", "follow_redirects", "raise_for_status")
    ), "Missing critical gotcha (timeout / follow_redirects / raise_for_status)"

    # Assertion 5: Boilerplate content from installation.md and contributing.md dropped
    installation_markers = [
        "HTTPX requires Python 3.8 or later",
        "python -m pip install httpx",
        "$ pip install httpx\n",
        "pip install httpx[cli]",  # legacy extra
    ]
    for marker in installation_markers:
        assert marker not in skill_md, f"Installation boilerplate leaked: {marker!r}"

    contributing_markers = [
        "## Contributing",
        "## Development",
        "git clone https://github.com/encode/httpx",
        "make install",
        "make test",
        "scripts/install",
        "scripts/test",
    ]
    for marker in contributing_markers:
        assert marker not in skill_md, f"Contributing boilerplate leaked: {marker!r}"

    assert not re.search(r"\bChangelog\b", skill_md, re.IGNORECASE), (
        "Changelog leaked into SKILL.md"
    )
    assert "bsd-3-clause" not in lower_md, "BSD-3-Clause license text leaked"
    assert not re.search(r"\bMIT License\b", skill_md, re.IGNORECASE), "MIT license leaked"

    # Assertion 6: Required sections present as ## headers
    for section in ("## API surface", "## Common workflows", "## Gotchas", "## Reference"):
        assert re.search(
            rf"^{re.escape(section)}\b", skill_md, re.MULTILINE
        ), f"Missing required section header: {section}"

    # Assertion 7: Uncondensed baseline exceeds 15k tokens (proves condensation compressed corpus)
    assert baseline_skill_tokens > 15_000, (
        f"Expected uncondensed baseline > 15,000 tokens, got {baseline_skill_tokens}"
    )

