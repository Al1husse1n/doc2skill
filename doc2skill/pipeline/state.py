import os
from typing import Any, TypedDict

from doc2skill.ingest.base import Document


class PipelineState(TypedDict, total=False):
    """Pipeline state schema for doc2skill LangGraph orchestration."""

    source_path: str
    documents: list[Document]
    distilled: str
    distilled_sources: list[str]
    dropped_documents: list[str]
    truncated: bool
    skill_md: str | None
    estimated_tokens: int
    actual_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int
    condense_input_tokens: int
    condense_output_tokens: int
    errors: list[str]
    warnings: list[str]
    max_tokens: int
    max_pages: int
    max_output_tokens: int
    max_context_tokens: int
    output_dir: str
    output_path: str | None
    frontmatter: dict[str, str] | None
    skip_llm: bool
    no_llm: bool
    no_condense: bool
    model: str
    llm_provider: Any
    validation_report: dict | None
    strict: bool


def create_initial_state(
    source_path: str,
    max_tokens: int = 50_000,
    output_dir: str = "./output",
    no_llm: bool = False,
    model: str | None = None,
    llm_provider: Any = None,
    documents: list[Document] | None = None,
    max_pages: int = 100,
    max_output_tokens: int = 15_000,
    max_context_tokens: int = 500_000,
    no_condense: bool = False,
    strict: bool = False,
) -> PipelineState:
    """Helper to initialize a fresh PipelineState dictionary."""
    selected_model = model or os.getenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")
    docs = list(documents) if documents else []
    return {
        "source_path": source_path,
        "documents": docs,
        "distilled": "",
        "distilled_sources": [doc.path for doc in docs],
        "dropped_documents": [],
        "truncated": False,
        "skill_md": None,
        "estimated_tokens": 0,
        "actual_tokens": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "condense_input_tokens": 0,
        "condense_output_tokens": 0,
        "errors": [],
        "warnings": [],
        "max_tokens": max_tokens,
        "max_pages": max_pages,
        "max_output_tokens": max_output_tokens,
        "max_context_tokens": max_context_tokens,
        "output_dir": output_dir,
        "output_path": None,
        "frontmatter": None,
        "skip_llm": False,
        "no_llm": no_llm,
        "no_condense": no_condense,
        "model": selected_model,
        "llm_provider": llm_provider,
        "validation_report": None,
        "strict": strict,
    }
