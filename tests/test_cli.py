from pathlib import Path

import pytest
from typer.testing import CliRunner

from doc2skill.cli import app

runner = CliRunner()


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
