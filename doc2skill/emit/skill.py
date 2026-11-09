import re
from collections.abc import Sequence
from pathlib import Path

from doc2skill.ingest.base import Document

DEFAULT_MAX_DESC_LENGTH = 200


def derive_skill_name(source_path: str | Path) -> str:
    """Derive skill name from a directory path (lowercase kebab-case)."""
    raw_name = Path(source_path).resolve().name.strip()
    if not raw_name or raw_name == ".":
        raw_name = "skill"
    # Convert spaces/underscores to hyphens and keep alphanumeric/hyphen
    slug = re.sub(r"[_\s]+", "-", raw_name)
    slug = re.sub(r"[^a-zA-Z0-9-]", "", slug).strip("-").lower()
    return slug or "skill"


def extract_description(
    documents: Sequence[Document],
    max_length: int = DEFAULT_MAX_DESC_LENGTH,
) -> str:
    """Extract the first non-empty paragraph from the first document, truncated."""
    if not documents:
        return "Skill generated from documentation."

    first_doc = documents[0]
    content = first_doc.content

    # Strip UTF-8 BOM if present
    text = content.lstrip("\ufeff").strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()

    # Split into paragraphs by double newlines
    paragraphs = text.split("\n\n")

    for block in paragraphs:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        # If block starts with markdown heading, extract any subsequent non-heading lines
        if lines[0].startswith("#"):
            non_heading = [l for l in lines if not l.startswith("#")]
            if not non_heading:
                continue
            paragraph_text = " ".join(non_heading).strip()
        else:
            paragraph_text = " ".join(lines).strip()

        if paragraph_text:
            if len(paragraph_text) > max_length:
                return paragraph_text[: max_length - 3].rstrip() + "..."
            return paragraph_text

    # Fallback if no paragraph found
    fallback_title = first_doc.title or "documentation"
    return f"Skill instructions and reference for {fallback_title}."


def render_skill_md(
    documents: Sequence[Document],
    skill_name: str | None = None,
    source_path: str | Path | None = None,
    frontmatter: dict[str, str] | None = None,
) -> str:
    """Generate SKILL.md markdown text from ingested documents and optional frontmatter."""
    if not skill_name:
        if source_path:
            skill_name = derive_skill_name(source_path)
        else:
            skill_name = "skill"

    if frontmatter and frontmatter.get("description"):
        description = frontmatter["description"].strip()
        when_to_use = frontmatter.get(
            "when_to_use", "(placeholder — will be LLM-generated later)"
        ).strip()
    else:
        description = extract_description(documents)
        when_to_use = "(placeholder — will be LLM-generated later)"

    doc_blocks: list[str] = []
    source_list: list[str] = []
    for doc in documents:
        clean_content = doc.content.strip()
        doc_blocks.append(f"### {doc.path}\n\n{clean_content}")
        source_list.append(f"- {doc.path}")

    reference_body = "\n\n---\n\n".join(doc_blocks)
    source_body = "\n".join(source_list)

    return f"""---
name: {skill_name}
description: {description}
---

# {skill_name}

## When to use
{when_to_use}

## Reference
{reference_body}

## Source
{source_body}
"""


def write_skill_file(
    skill_md: str,
    skill_name: str,
    output_dir: str | Path = "./output",
) -> Path:
    """Write rendered SKILL.md to <output_dir>/<skill_name>/SKILL.md."""
    target_dir = Path(output_dir) / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "SKILL.md"
    out_file.write_text(skill_md, encoding="utf-8")
    return out_file


def emit_skill(
    documents: Sequence[Document],
    skill_name: str | None = None,
    source_path: str | Path | None = None,
    output_dir: str | Path = "./output",
    frontmatter: dict[str, str] | None = None,
) -> tuple[str, Path]:
    """Render and write SKILL.md for given documents and frontmatter."""
    if not skill_name:
        if source_path:
            skill_name = derive_skill_name(source_path)
        else:
            skill_name = "skill"

    skill_md = render_skill_md(
        documents,
        skill_name=skill_name,
        source_path=source_path,
        frontmatter=frontmatter,
    )
    out_path = write_skill_file(skill_md, skill_name, output_dir=output_dir)
    return skill_md, out_path
