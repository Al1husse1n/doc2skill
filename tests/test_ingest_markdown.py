from pathlib import Path

import pytest

from doc2skill.ingest.markdown import extract_title, load_markdown_directory


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Fixture directory with 3 valid markdown files and ignored directories."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir()

    # File 1: Markdown with standard # heading
    file1 = docs_dir / "01_intro.md"
    file1.write_text(
        "# Introduction to Doc2Skill\n\nDoc2Skill converts documentation to skills.",
        encoding="utf-8",
    )

    # File 2: MDX with # heading
    file2 = docs_dir / "02_guide.mdx"
    file2.write_text(
        "# User Guide\n\nStep by step guide on using the tool.",
        encoding="utf-8",
    )

    # File 3: Markdown without heading (fallback to filename)
    file3 = docs_dir / "03_reference.md"
    file3.write_text(
        "This file contains reference notes without an H1 heading.",
        encoding="utf-8",
    )

    # Ignored directories should be skipped
    node_modules = docs_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "skip.md").write_text("# Ignored Node Module", encoding="utf-8")

    git_dir = docs_dir / ".git"
    git_dir.mkdir()
    (git_dir / "skip.md").write_text("# Ignored Git", encoding="utf-8")

    venv_dir = docs_dir / ".venv"
    venv_dir.mkdir()
    (venv_dir / "skip.md").write_text("# Ignored Venv", encoding="utf-8")

    return docs_dir


def test_ingest_sample_directory(sample_docs_dir: Path):
    """Assert loading a directory with 3 markdown files produces 3 Documents with correct titles."""
    documents = load_markdown_directory(sample_docs_dir)

    assert len(documents) == 3

    assert documents[0].path == "01_intro.md"
    assert documents[0].title == "Introduction to Doc2Skill"
    assert "Doc2Skill converts" in documents[0].content

    assert documents[1].path == "02_guide.mdx"
    assert documents[1].title == "User Guide"
    assert "Step by step" in documents[1].content

    assert documents[2].path == "03_reference.md"
    assert documents[2].title == "03_reference"
    assert "reference notes" in documents[2].content


def test_extract_title_helper():
    """Test title extraction and fallbacks."""
    assert extract_title("# My Custom Title\n\nbody", "file.md") == "My Custom Title"
    assert extract_title("## Only H2\n\nbody", "fallback_name.md") == "fallback_name"
    assert extract_title("No headings at all", "my_doc.md") == "my_doc"
