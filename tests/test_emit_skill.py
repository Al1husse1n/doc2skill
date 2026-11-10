from pathlib import Path

from doc2skill.emit.skill import (
    FALLBACK_WHEN_TO_USE,
    derive_skill_name,
    emit_skill,
    extract_description,
    format_raw_documents_body,
    render,
)
from doc2skill.ingest.base import Document


def test_render_with_synthesized_frontmatter_and_condensed_body():
    """Verify render() formats frontmatter, Title Case heading, when_to_use, verbatim body, and sources."""
    body = (
        "## API surface\n`Client.get(url)`\n\n"
        "## Common workflows\n1. Create client.\n2. Send request.\n\n"
        "## Gotchas\nAlways close async clients.\n\n"
        "## Reference\nSee client API."
    )
    sources = ["https://docs.example.com/client", "https://docs.example.com/async"]
    frontmatter = {
        "description": "HTTP client library skill for sync and async requests.",
        "when_to_use": "- When making HTTP requests in Python\n- When configuring timeouts or retries",
    }

    skill_name, skill_md = render(
        body=body,
        sources=sources,
        frontmatter=frontmatter,
        source_path="docs/my_http_lib",
    )

    assert skill_name == "my-http-lib"
    assert skill_md.startswith("---\nname: my-http-lib\n")
    assert "description: HTTP client library skill for sync and async requests.\n---\n" in skill_md
    assert "# My Http Lib\n" in skill_md

    # Frontmatter when_to_use renders BEFORE body, and body passes through verbatim without double When to use
    when_pos = skill_md.index("## When to use\n- When making HTTP requests")
    api_pos = skill_md.index("## API surface\n`Client.get(url)`")
    assert when_pos < api_pos
    assert skill_md.count("## When to use") == 1
    assert body in skill_md

    # ## Source section renders all sources as bullets
    assert "## Source\n\n- https://docs.example.com/client\n- https://docs.example.com/async" in skill_md


def test_emit_skill_fallback_frontmatter_and_writes_file(tmp_path: Path):
    """Verify emit_skill writes SKILL.md with fallback frontmatter when frontmatter is None."""
    docs = [
        Document(
            path="guides/quickstart.md",
            content="# Quickstart\n\nThis guide shows you how to quickly get started with the tool.",
            title="Quickstart",
        ),
        Document(
            path="api/endpoints.md",
            content="# API Endpoints\n\nDetailed specifications of all available endpoints.",
            title="API Endpoints",
        ),
    ]
    raw_body = format_raw_documents_body(docs)
    sources = [d.path for d in docs]

    skill_md, output_file = emit_skill(
        body=raw_body,
        sources=sources,
        frontmatter=None,
        source_path="docs/test_skill",
        output_dir=tmp_path,
        fallback_description=extract_description(docs),
    )

    expected_path = tmp_path / "test-skill" / "SKILL.md"
    assert output_file == expected_path
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8") == skill_md

    assert "name: test-skill\n" in skill_md
    assert "description: This guide shows you how to quickly get started with the tool.\n" in skill_md
    assert f"## When to use\n{FALLBACK_WHEN_TO_USE}\n" in skill_md
    assert "### guides/quickstart.md" in skill_md
    assert "### api/endpoints.md" in skill_md
    assert "## Source\n\n- guides/quickstart.md\n- api/endpoints.md" in skill_md


def test_derive_skill_name():
    """Test skill name derivation from directory paths and URLs."""
    assert derive_skill_name("docs/my_cool_project") == "my-cool-project"
    assert derive_skill_name("/path/to/some-tool") == "some-tool"
    assert derive_skill_name("https://example.com/") == "example"
    assert derive_skill_name(".") != ""


def test_extract_description_truncation():
    """Test description extraction and truncation to max length."""
    long_text = "Word " * 60
    doc = Document(path="doc.md", content=f"# Title\n\n{long_text}", title="Title")
    desc = extract_description([doc], max_length=100)
    assert len(desc) <= 100
    assert desc.endswith("...")
