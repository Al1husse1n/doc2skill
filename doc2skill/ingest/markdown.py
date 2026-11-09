import re
from pathlib import Path

from doc2skill.ingest.base import Document

IGNORED_DIRS = {"node_modules", ".git", ".venv"}


def extract_title(content: str, filename: str) -> str:
    """Extract the first '#' heading from markdown content, or fallback to filename stem."""
    # Match ATX-style top-level heading: '# Heading Text'
    for line in content.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            title = match.group(1).strip()
            if title:
                return title

    # Fallback to filename without extension
    return Path(filename).stem


def load_markdown_directory(source_path: str | Path) -> list[Document]:
    """Recursively load .md and .mdx files from directory, sorted by relative path.

    Skips directories named node_modules, .git, and .venv.
    """
    root = Path(source_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Source path '{source_path}' is not a valid directory.")

    documents: list[Document] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in (".md", ".mdx"):
            continue

        rel_path = path.relative_to(root)
        # Check if any part of relative path is in IGNORED_DIRS
        if any(part in IGNORED_DIRS for part in rel_path.parts):
            continue

        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8-sig", errors="replace")

        # Use forward slashes for cross-platform consistency
        display_path = rel_path.as_posix()
        title = extract_title(content, path.name)

        documents.append(
            Document(
                path=display_path,
                content=content,
                title=title,
                metadata={"absolute_path": str(path)},
            )
        )

    documents.sort(key=lambda d: d.path)
    return documents
