from pathlib import Path

import pytest
from typer.testing import CliRunner

from doc2skill.cli import app

runner = CliRunner()


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Typer CliRunner."""
    return CliRunner()


@pytest.fixture
def cli_docs_dir(tmp_path: Path) -> Path:
    """Fixture directory for CLI testing."""
    docs = tmp_path / "cli_sample"
    docs.mkdir()
    (docs / "main.md").write_text(
        "# CLI Sample Docs\n\nTesting the doc2skill command line interface.",
        encoding="utf-8",
    )
    return docs


def test_cli_execution_and_caching(
    cli_docs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test CLI run, summary output, cache population, and cache hit."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(cache_dir))

    # 1. First run: cache miss, runs pipeline
    result1 = runner.invoke(
        app,
        [str(cli_docs_dir), "--output", str(output_dir), "--no-llm"],
    )
    assert result1.exit_code == 0
    assert "Summary:" in result1.output
    assert "Documents loaded: 1" in result1.output
    assert "Estimated input tokens:" in result1.output
    assert "Output:" in result1.output
    assert "Cache hit" not in result1.output

    expected_skill = output_dir / "cli-sample" / "SKILL.md"
    assert expected_skill.exists()

    # 2. Second run without --no-cache: cache hit
    result2 = runner.invoke(
        app,
        [str(cli_docs_dir), "--output", str(output_dir), "--no-llm"],
    )
    assert result2.exit_code == 0
    assert "Cache hit: writing from cache" in result2.output
    assert "Summary:" in result2.output

    # 3. Third run with --no-cache: runs pipeline again
    result3 = runner.invoke(
        app,
        [str(cli_docs_dir), "--output", str(output_dir), "--no-llm", "--no-cache"],
    )
    assert result3.exit_code == 0
    assert "Cache hit" not in result3.output
    assert "Documents loaded: 1" in result3.output


def test_cli_budget_exceeded_fallback(
    cli_docs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test CLI prints LLM skipped (budget) and emits with fallback when budget exceeded."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(cache_dir))

    result = runner.invoke(
        app,
        [
            str(cli_docs_dir),
            "--output",
            str(output_dir),
            "--max-tokens",
            "1",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0
    assert "LLM skipped (budget). Used fallback frontmatter." in result.output
    assert "Summary:" in result.output

    emitted_skill = output_dir / "cli-sample" / "SKILL.md"
    assert emitted_skill.exists()
    content = emitted_skill.read_text(encoding="utf-8")
    assert "(placeholder — will be LLM-generated later)" in content


def test_cli_no_llm_flag(
    cli_docs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test CLI --no-llm flag forces fallback frontmatter without errors."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(cache_dir))

    result = runner.invoke(
        app,
        [
            str(cli_docs_dir),
            "--output",
            str(output_dir),
            "--no-llm",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0
    assert "LLM skipped (--no-llm). Used fallback frontmatter." in result.output
    assert "Documents loaded: 1" in result.output


def test_summary_printed_once(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Assert summary metrics are printed exactly once when pipeline runs."""
    fixture_dir = tmp_path / "summary_fixture"
    fixture_dir.mkdir()
    (fixture_dir / "index.md").write_text("# Doc\nSome text content.", encoding="utf-8")

    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path / "cache"))

    # Stub LLM to return deterministic frontmatter
    from unittest.mock import patch
    from doc2skill.llm.base import FrontmatterResult

    stub_frontmatter = FrontmatterResult(
        description="Deterministic skill description.",
        when_to_use="- When deterministic",
        input_tokens=80,
        output_tokens=25,
    )

    with patch(
        "doc2skill.pipeline.graph.GeminiProvider.synthesize_frontmatter",
        return_value=stub_frontmatter,
    ):
        result = cli_runner.invoke(app, ["build", str(fixture_dir), "--no-cache"])

    assert result.exit_code == 0
    assert result.output.count("Estimated input tokens") == 1
    assert result.output.count("Actual tokens used") == 1


def test_cli_summary_prints_estimated_tokens_exactly_once(
    cli_docs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that 'Estimated input tokens' appears exactly once in CLI output."""
    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(
        app,
        [str(cli_docs_dir), "--output", str(output_dir), "--no-llm", "--no-cache"],
    )
    assert result.exit_code == 0
    # Assert 'Estimated input tokens' appears exactly once
    assert result.output.count("Estimated input tokens") == 1


def test_cli_url_single_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify CLI accepts URL source and outputs single-page summary."""
    from unittest.mock import AsyncMock, patch

    from doc2skill.ingest.base import Document

    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path / "cache"))

    mock_doc = Document(
        path="https://example.com/guide",
        content="# Web Guide\nSome useful content.",
        title="Web Guide",
    )

    with patch(
        "doc2skill.cli.load_url", new_callable=AsyncMock, return_value=[mock_doc]
    ):
        result = runner.invoke(
            app,
            [
                "https://example.com/guide",
                "--output",
                str(output_dir),
                "--no-llm",
                "--no-cache",
                "--no-sitemap",
            ],
        )

    assert result.exit_code == 0
    assert "Source: https://example.com/guide (single page)" in result.output
    assert "Documents loaded: 1" in result.output
    assert result.output.count("Estimated input tokens") == 1
    assert (output_dir / "guide" / "SKILL.md").exists()


def test_cli_url_sitemap_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify CLI with sitemap outputs sitemap page count in summary."""
    from unittest.mock import AsyncMock, patch

    from doc2skill.ingest.base import Document

    output_dir = tmp_path / "output"
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path / "cache"))

    docs = [
        Document(
            path="https://example.com/p1", content="# P1\nC1", title="Page 1"
        ),
        Document(
            path="https://example.com/p2", content="# P2\nC2", title="Page 2"
        ),
    ]

    with patch(
        "doc2skill.cli.load_url", new_callable=AsyncMock, return_value=docs
    ):
        result = runner.invoke(
            app,
            [
                "https://example.com/docs",
                "--output",
                str(output_dir),
                "--no-llm",
                "--no-cache",
                "--max-pages",
                "50",
            ],
        )

    assert result.exit_code == 0
    assert "Source: https://example.com/docs (sitemap, 2 pages)" in result.output
    assert "Documents loaded: 2" in result.output
    assert result.output.count("Estimated input tokens") == 1
