import os
from typing import Any, TypedDict

from doc2skill.ingest.base import Document


class PipelineState(TypedDict, total=False):
    """Pipeline state schema for doc2skill LangGraph orchestration."""

    source_path: str
    documents: list[Document]
    distilled: list[Document]
    skill_md: str | None
    estimated_tokens: int
    actual_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int
    errors: list[str]
    warnings: list[str]
    max_tokens: int
    output_dir: str
    output_path: str | None
    frontmatter: dict[str, str] | None
    skip_llm: bool
    no_llm: bool
    model: str
    llm_provider: Any


def create_initial_state(
    source_path: str,
    max_tokens: int = 50_000,
    output_dir: str = "./output",
    no_llm: bool = False,
    model: str | None = None,
    llm_provider: Any = None,
    documents: list[Document] | None = None,
) -> PipelineState:
    """Helper to initialize a fresh PipelineState dictionary."""
    selected_model = model or os.getenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")
    docs = list(documents) if documents else []
    return {
        "source_path": source_path,
        "documents": docs,
        "distilled": list(docs),
        "skill_md": None,
        "estimated_tokens": 0,
        "actual_tokens": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "errors": [],
        "warnings": [],
        "max_tokens": max_tokens,
        "output_dir": output_dir,
        "output_path": None,
        "frontmatter": None,
        "skip_llm": False,
        "no_llm": no_llm,
        "model": selected_model,
        "llm_provider": llm_provider,
    }
