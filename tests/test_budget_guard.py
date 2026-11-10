from pathlib import Path

import pytest

from doc2skill.budget.guard import (
    TokenBudget,
    estimate_documents_tokens,
    estimate_tokens,
)
from doc2skill.ingest.base import Document
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


@pytest.fixture
def docs_directory(tmp_path: Path) -> Path:
    """Create a temporary directory with markdown documentation."""
    docs = tmp_path / "project_docs"
    docs.mkdir()
    doc1 = docs / "overview.md"
    doc1.write_text(
        "# Project Overview\n\nThis project provides extensive documentation for skills.",
        encoding="utf-8",
    )
    return docs


def test_pipeline_budget_guard_exceeded(docs_directory: Path, tmp_path: Path):
    """Pipeline with max_tokens=1 returns an error in state and skips LLM."""
    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path=str(docs_directory),
        max_tokens=1,
        output_dir=str(tmp_path / "output"),
    )

    result = graph.invoke(initial_state)

    assert len(result["errors"]) > 0
    assert any("Budget exceeded" in err for err in result["errors"])
    # File is emitted using fallback frontmatter
    assert result.get("output_path") is not None
    assert Path(result["output_path"]).exists()
    assert "(placeholder — will be LLM-generated later)" in result["skill_md"]


def test_pipeline_budget_guard_within_limit(docs_directory: Path, tmp_path: Path):
    """Pipeline with default max_tokens succeeds without errors."""
    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path=str(docs_directory),
        max_tokens=50_000,
        output_dir=str(tmp_path / "output"),
        no_llm=True,
    )

    result = graph.invoke(initial_state)

    assert len(result["errors"]) == 0
    assert result["estimated_tokens"] > 0
    assert result["skill_md"] is not None
    assert result["output_path"] is not None
    assert Path(result["output_path"]).exists()


def test_token_budget_unit():
    """Unit tests for TokenBudget class and estimation helpers."""
    text = "Hello world! This is a test."
    tokens = estimate_tokens(text)
    assert tokens > 0

    doc1 = Document(path="a.md", content="Four words here now.", title="A")
    doc2 = Document(path="b.md", content="More words here too.", title="B")
    total_tokens = estimate_documents_tokens([doc1, doc2])
    assert total_tokens >= 2

    budget = TokenBudget(max_tokens=1)
    budget.estimate([doc1, doc2])
    assert budget.is_exceeded()
    assert budget.check() is not None

    generous_budget = TokenBudget(max_tokens=1000)
    generous_budget.estimate([doc1, doc2])
    assert not generous_budget.is_exceeded()
    assert generous_budget.check() is None
