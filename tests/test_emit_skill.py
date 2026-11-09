from pathlib import Path

import pytest

from doc2skill.emit.skill import (
    derive_skill_name,
    emit_skill,
    extract_description,
)
from doc2skill.ingest.base import Document


@pytest.fixture
def two_fixture_documents() -> list[Document]:
    """Fixture containing 2 Document objects."""
    doc1 = Document(
        path="guides/quickstart.md",
        content="# Quickstart\n\nThis guide shows you how to quickly get started with the tool.\n\nEnjoy!",
        title="Quickstart",
    )
    doc2 = Document(
        path="api/endpoints.md",
        content="# API Endpoints\n\nDetailed specifications of all available endpoints.",
        title="API Endpoints",
    )
    return [doc1, doc2]


def test_emit_from_two_fixture_documents(
    tmp_path: Path, two_fixture_documents: list[Document]
):
    """Emit from 2 fixture Documents, assert SKILL.md contains expected frontmatter and content."""
    skill_name = "test-skill"
    skill_md, output_file = emit_skill(
        two_fixture_documents,
        skill_name=skill_name,
        output_dir=tmp_path,
    )

    # 1. Output file assertion
    expected_path = tmp_path / "test-skill" / "SKILL.md"
    assert output_file == expected_path
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8") == skill_md

    # 2. Frontmatter assertions
    assert skill_md.startswith("---\n")
    assert "name: test-skill\n" in skill_md
    assert (
        "description: This guide shows you how to quickly get started with the tool.\n"
        in skill_md
    )
    assert "---\n\n# test-skill\n" in skill_md

    # 3. Content assertions
    assert "## When to use\n(placeholder — will be LLM-generated later)\n" in skill_md
    assert "## Reference\n" in skill_md

    # Check both document paths and contents are present
    assert "### guides/quickstart.md" in skill_md
    assert "This guide shows you how to quickly get started with the tool." in skill_md
    assert "### api/endpoints.md" in skill_md
    assert "Detailed specifications of all available endpoints." in skill_md

    # Check separator between documents
    assert "\n\n---\n\n" in skill_md
    # Check source section
    assert "## Source\n- guides/quickstart.md\n- api/endpoints.md" in skill_md


def test_emit_with_synthesized_frontmatter(
    tmp_path: Path, two_fixture_documents: list[Document]
):
    """Assert emit_skill uses synthesized frontmatter when provided."""
    frontmatter = {
        "description": "Synthesized API skill description.",
        "when_to_use": "- When integrating APIs\n- When querying endpoints",
    }
    skill_md, _output_file = emit_skill(
        two_fixture_documents,
        skill_name="synth-skill",
        output_dir=tmp_path,
        frontmatter=frontmatter,
    )
    assert "description: Synthesized API skill description." in skill_md
    assert (
        "## When to use\n- When integrating APIs\n- When querying endpoints" in skill_md
    )
    assert "(placeholder — will be LLM-generated later)" not in skill_md


def test_derive_skill_name():
    """Test skill name derivation from directory paths."""
    assert derive_skill_name("docs/my_cool_project") == "my-cool-project"
    assert derive_skill_name("/path/to/some-tool") == "some-tool"
    assert derive_skill_name(".") != ""


def test_extract_description_truncation():
    """Test description extraction and truncation to max length."""
    long_text = "Word " * 60  # ~300 chars
    doc = Document(path="doc.md", content=f"# Title\n\n{long_text}", title="Title")
    desc = extract_description([doc], max_length=100)
    assert len(desc) <= 100
    assert desc.endswith("...")
