from langgraph.graph import END, START, StateGraph

from doc2skill.budget.guard import (
    DEFAULT_MAX_TOKENS,
    TokenBudget,
    estimate_documents_tokens,
)
from doc2skill.distill.stub import distill_passthrough
from doc2skill.emit.skill import derive_skill_name, render_skill_md, write_skill_file
from doc2skill.ingest.markdown import load_markdown_directory
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

    estimated = estimate_documents_tokens(documents)
    return {
        "documents": documents,
        "estimated_tokens": estimated,
        "errors": errors,
    }


def distill_node(state: PipelineState) -> dict:
    """Pass-through stub for document distillation."""
    documents = state.get("documents", [])
    distilled = distill_passthrough(documents)
    return {"distilled": distilled}


def budget_guard_node(state: PipelineState) -> dict:
    """Check estimated tokens against configurable max_tokens."""
    estimated = state.get("estimated_tokens", 0)
    if estimated == 0:
        docs = state.get("distilled") or state.get("documents", [])
        estimated = estimate_documents_tokens(docs)

    max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    budget = TokenBudget(
        max_tokens=max_tokens,
        estimated_tokens=estimated,
        actual_tokens=state.get("actual_tokens", 0),
    )

    errors = list(state.get("errors", []))
    error_msg = budget.check()
    if error_msg:
        errors.append(error_msg)

    return {
        "estimated_tokens": estimated,
        "errors": errors,
    }


def emit_skill_node(state: PipelineState) -> dict:
    """Render and write SKILL.md output."""
    source_path = state.get("source_path", "")
    documents = state.get("distilled") or state.get("documents", [])
    skill_name = derive_skill_name(source_path) if source_path else "skill"

    skill_md = render_skill_md(
        documents, skill_name=skill_name, source_path=source_path
    )

    # If errors occurred in prior stages, do not write output to disk
    if state.get("errors"):
        return {"skill_md": skill_md, "output_path": None}

    output_dir = state.get("output_dir", "./output")
    out_file = write_skill_file(skill_md, skill_name, output_dir=output_dir)
    return {
        "skill_md": skill_md,
        "output_path": str(out_file),
    }


def build_pipeline_graph():
    """Build and compile the linear LangGraph pipeline:

    load_documents -> distill -> budget_guard -> emit_skill
    """
    workflow = StateGraph(PipelineState)

    workflow.add_node("load_documents", load_documents_node)
    workflow.add_node("distill", distill_node)
    workflow.add_node("budget_guard", budget_guard_node)
    workflow.add_node("emit_skill", emit_skill_node)

    workflow.add_edge(START, "load_documents")
    workflow.add_edge("load_documents", "distill")
    workflow.add_edge("distill", "budget_guard")
    workflow.add_edge("budget_guard", "emit_skill")
    workflow.add_edge("emit_skill", END)

    return workflow.compile()
