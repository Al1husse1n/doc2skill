import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from doc2skill.ingest.base import Document

DEFAULT_MAX_DESC_LENGTH = 200
FALLBACK_WHEN_TO_USE = "(placeholder — will be LLM-generated later)"


def derive_skill_name(source_path: str | Path) -> str:
    """Derive skill name from a directory path or URL (lowercase kebab-case)."""
    src_str = str(source_path).strip()
    if src_str.startswith(("http://", "https://")):
        parsed = urlparse(src_str)
        clean_path = parsed.path.strip("/")
        if clean_path and not clean_path.endswith(".xml"):
            raw_name = Path(clean_path).name or clean_path
        else:
            host = parsed.netloc.split(":")[0].replace("www.", "")
            parts = host.split(".")
            raw_name = parts[0] if len(parts) > 1 else host
    else:
        raw_name = Path(source_path).resolve().name.strip()

    if not raw_name or raw_name == ".":
        raw_name = "skill"
    slug = re.sub(r"[_\s.]+", "-", raw_name)
    slug = re.sub(r"[^a-zA-Z0-9-]", "", slug).strip("-").lower()
    return slug or "skill"


def format_skill_title(skill_name: str) -> str:
    """Convert kebab-case skill slug into Title Case heading."""
    parts = [part.capitalize() for part in skill_name.split("-") if part]
    return " ".join(parts) if parts else "Skill"


def extract_description(
    documents: Sequence[Document],
    max_length: int = DEFAULT_MAX_DESC_LENGTH,
) -> str:
    """Extract the first non-empty paragraph from the first document, truncated."""
    if not documents:
        return "Skill generated from documentation."

    first_doc = documents[0]
    content = first_doc.content

    text = content.lstrip("\ufeff").strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()

    paragraphs = text.split("\n\n")

    for block in paragraphs:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if lines[0].startswith("#"):
            non_heading = [line for line in lines if not line.startswith("#")]
            if not non_heading:
                continue
            paragraph_text = " ".join(non_heading).strip()
        else:
            paragraph_text = " ".join(lines).strip()

        if paragraph_text:
            if len(paragraph_text) > max_length:
                return paragraph_text[: max_length - 3].rstrip() + "..."
            return paragraph_text

    fallback_title = first_doc.title or "documentation"
    return f"Skill instructions and reference for {fallback_title}."


def format_raw_documents_body(documents: Sequence[Document]) -> str:
    """Render raw documents into v4 Reference concatenation for fallback / --no-condense."""
    if not documents:
        return "## Reference"
    doc_blocks = [
        f"### {doc.path}\n\n{doc.content.strip()}" for doc in documents
    ]
    reference_body = "\n\n---\n\n".join(doc_blocks)
    return f"## Reference\n{reference_body}"


def render(
    body: str,
    sources: Sequence[str],
    frontmatter: dict[str, str] | None,
    source_path: str | Path,
    skill_name: str | None = None,
    fallback_description: str | None = None,
) -> tuple[str, str]:
    """Render final SKILL.md markdown and return (skill_name, skill_md)."""
    resolved_name = skill_name or (derive_skill_name(source_path) if source_path else "skill")
    title = format_skill_title(resolved_name)

    if frontmatter and frontmatter.get("description"):
        description = frontmatter["description"].strip()
        when_to_use = frontmatter.get("when_to_use", FALLBACK_WHEN_TO_USE).strip()
    else:
        description = fallback_description or "Skill generated from documentation."
        when_to_use = FALLBACK_WHEN_TO_USE

    clean_body = body.strip()
    source_lines = "\n".join(f"- {src}" for src in sources)

    skill_md = f"""---
name: {resolved_name}
description: {description}
---

# {title}

## When to use
{when_to_use}

{clean_body}

## Source

{source_lines}
"""
    return resolved_name, skill_md


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
    body: str,
    sources: Sequence[str],
    frontmatter: dict[str, str] | None = None,
    source_path: str | Path = "",
    output_dir: str | Path = "./output",
    skill_name: str | None = None,
    fallback_description: str | None = None,
) -> tuple[str, Path]:
    """Render and write SKILL.md for condensed body, sources, and frontmatter."""
    resolved_name, skill_md = render(
        body=body,
        sources=sources,
        frontmatter=frontmatter,
        source_path=source_path,
        skill_name=skill_name,
        fallback_description=fallback_description,
    )
    out_path = write_skill_file(skill_md, resolved_name, output_dir=output_dir)
    return skill_md, out_path
