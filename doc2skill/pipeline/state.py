from typing import TypedDict

from doc2skill.ingest.base import Document


class PipelineState(TypedDict, total=False):
    """Pipeline state schema for doc2skill LangGraph orchestration."""

    source_path: str
    documents: list[Document]
    distilled: list[Document]
    skill_md: str | None
    estimated_tokens: int
    actual_tokens: int
    errors: list[str]
    max_tokens: int
    output_dir: str
    output_path: str | None


def create_initial_state(
    source_path: str,
    max_tokens: int = 50_000,
    output_dir: str = "./output",
) -> PipelineState:
    """Helper to initialize a fresh PipelineState dictionary."""
    return {
        "source_path": source_path,
        "documents": [],
        "distilled": [],
        "skill_md": None,
        "estimated_tokens": 0,
        "actual_tokens": 0,
        "errors": [],
        "max_tokens": max_tokens,
        "output_dir": output_dir,
        "output_path": None,
    }
