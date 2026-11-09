import os
from typing import Literal

from google.genai import errors as genai_errors
from langgraph.graph import END, START, StateGraph

from doc2skill.budget.guard import DEFAULT_MAX_TOKENS, estimate_prompt_tokens
from doc2skill.emit.skill import derive_skill_name, render_skill_md, write_skill_file
from doc2skill.ingest.markdown import load_markdown_directory
from doc2skill.llm.gemini import GeminiProvider
from doc2skill.pipeline.state import PipelineState


def load_documents_node(state: PipelineState) -> dict:
    """Load markdown documents from state source_path."""
    source_path = state.get("source_path", "")
    errors = list(state.get("errors", []))

    try:
        documents = load_markdown_directory(source_path)
    except (OSError, ValueError) as exc:
        errors.append(f"Failed to load documents: {exc}")
        return {"documents": [], "errors": errors, "estimated_tokens": 0}

    return {
        "documents": documents,
        "distilled": list(documents),
        "errors": errors,
    }


def budget_guard_node(state: PipelineState) -> dict:
    """Pre-check input token count before making any LLM calls."""
    documents = state.get("documents", [])
    max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    no_llm = state.get("no_llm", False)
    provider = state.get("llm_provider")

    if provider is None and not no_llm and os.environ.get("GEMINI_API_KEY"):
        try:
            provider = GeminiProvider(model=state.get("model", "gemini-2.5-flash"))
        except (RuntimeError, ValueError, OSError):
            provider = None

    estimated = estimate_prompt_tokens(documents, provider=provider)
    errors = list(state.get("errors", []))

    if no_llm:
        return {
            "estimated_tokens": estimated,
            "skip_llm": True,
            "llm_provider": provider,
        }

    # Pre-call budget enforcement
    if estimated > max_tokens:
        errors.append(
            f"Budget exceeded: estimated {estimated} input tokens > max {max_tokens}. LLM call skipped."
        )
        return {
            "estimated_tokens": estimated,
            "errors": errors,
            "skip_llm": True,
            "llm_provider": provider,
        }

    if provider is None and not os.environ.get("GEMINI_API_KEY"):
        return {
            "estimated_tokens": estimated,
            "skip_llm": True,
            "llm_provider": None,
        }

    return {
        "estimated_tokens": estimated,
        "skip_llm": False,
        "llm_provider": provider,
    }


def should_distill(state: PipelineState) -> Literal["distill", "skip"]:
    """Conditional router: skip LLM distill if budget exceeded or no-llm requested."""
    if state.get("skip_llm") or state.get("no_llm"):
        return "skip"
    # Skip if errors other than budget exceeded exist
    if any("Failed to load documents" in err for err in state.get("errors", [])):
        return "skip"
    return "distill"


def distill_node(state: PipelineState) -> dict:
    """Invoke LLM provider to synthesize frontmatter metadata."""
    documents = state.get("documents", [])
    provider = state.get("llm_provider")
    errors = list(state.get("errors", []))

    if provider is None:
        try:
            provider = GeminiProvider(model=state.get("model", "gemini-2.5-flash"))
        except (RuntimeError, ValueError, OSError) as exc:
            errors.append(f"Failed to initialize LLM provider: {exc}")
            return {
                "distilled": list(documents),
                "frontmatter": None,
                "errors": errors,
            }

    try:
        result = provider.synthesize_frontmatter(
            documents, max_tokens=state.get("max_tokens", 1000)
        )
        actual_tokens = result.input_tokens + result.output_tokens
        frontmatter = {
            "description": result.description,
            "when_to_use": result.when_to_use,
        }
        return {
            "distilled": list(documents),
            "frontmatter": frontmatter,
            "actual_tokens": actual_tokens,
            "llm_input_tokens": result.input_tokens,
            "llm_output_tokens": result.output_tokens,
        }
    except (genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
        errors.append(f"LLM synthesis failed: {exc}")
        return {
            "distilled": list(documents),
            "frontmatter": None,
            "errors": errors,
        }


def budget_record_node(state: PipelineState) -> dict:
    """Post-call accounting node (checks if actual spend exceeded max_tokens)."""
    actual = state.get("actual_tokens", 0)
    max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    warnings = list(state.get("warnings", []))

    if actual > max_tokens:
        warnings.append(
            f"Budget warning: actual {actual} tokens exceeded maximum allowed {max_tokens}."
        )

    distilled = state.get("distilled") or state.get("documents", [])
    return {"warnings": warnings, "distilled": distilled}


def emit_skill_node(state: PipelineState) -> dict:
    """Render and write SKILL.md output using synthesized or fallback frontmatter."""
    source_path = state.get("source_path", "")
    documents = state.get("distilled") or state.get("documents", [])
    skill_name = derive_skill_name(source_path) if source_path else "skill"
    frontmatter = state.get("frontmatter")

    skill_md = render_skill_md(
        documents,
        skill_name=skill_name,
        source_path=source_path,
        frontmatter=frontmatter,
    )

    output_dir = state.get("output_dir", "./output")
    out_file = write_skill_file(skill_md, skill_name, output_dir=output_dir)
    return {
        "skill_md": skill_md,
        "output_path": str(out_file),
    }


def build_pipeline_graph():
    """Build and compile the LangGraph pipeline:

    load_documents -> budget_guard -> [distill ->] budget_record -> emit_skill
    """
    workflow = StateGraph(PipelineState)

    workflow.add_node("load_documents", load_documents_node)
    workflow.add_node("budget_guard", budget_guard_node)
    workflow.add_node("distill", distill_node)
    workflow.add_node("budget_record", budget_record_node)
    workflow.add_node("emit_skill", emit_skill_node)

    workflow.add_edge(START, "load_documents")
    workflow.add_edge("load_documents", "budget_guard")
    workflow.add_conditional_edges(
        "budget_guard",
        should_distill,
        {
            "distill": "distill",
            "skip": "budget_record",
        },
    )
    workflow.add_edge("distill", "budget_record")
    workflow.add_edge("budget_record", "emit_skill")
    workflow.add_edge("emit_skill", END)

    return workflow.compile()
