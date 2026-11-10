import asyncio
import concurrent.futures
import inspect
import os
from typing import Literal

from google.genai import errors as genai_errors
from langgraph.graph import END, START, StateGraph

from doc2skill.budget.guard import (
    DEFAULT_MAX_TOKENS,
    estimate_condense_tokens,
    estimate_prompt_tokens,
)
from doc2skill.distill.condense import fit_documents_to_context
from doc2skill.emit.skill import (
    derive_skill_name,
    extract_description,
    format_raw_documents_body,
    render,
    write_skill_file,
)
from doc2skill.ingest.markdown import load_markdown_directory
from doc2skill.llm.gemini import DEFAULT_MODEL, GeminiProvider
from doc2skill.pipeline.state import PipelineState


def _run_async(coro):
    """Run an async coroutine from a sync node regardless of active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def load_documents_node(state: PipelineState) -> dict:
    """Load markdown documents from state source_path or keep preloaded documents."""
    preloaded = state.get("documents")
    if preloaded:
        return {
            "documents": preloaded,
            "distilled": "",
            "distilled_sources": [doc.path for doc in preloaded],
            "dropped_documents": [],
            "truncated": False,
            "errors": list(state.get("errors", [])),
        }

    source_path = state.get("source_path", "")
    errors = list(state.get("errors", []))

    if source_path.startswith(("http://", "https://")):
        try:
            from doc2skill.ingest.url import load_url

            max_pages = state.get("max_pages", 100)
            documents = _run_async(load_url(source_path, max_pages=max_pages))
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"Failed to load URL documents: {exc}")
            return {"documents": [], "errors": errors, "estimated_tokens": 0}
    else:
        try:
            documents = load_markdown_directory(source_path)
        except (OSError, ValueError) as exc:
            errors.append(f"Failed to load documents: {exc}")
            return {"documents": [], "errors": errors, "estimated_tokens": 0}

    return {
        "documents": documents,
        "distilled": "",
        "distilled_sources": [doc.path for doc in documents],
        "dropped_documents": [],
        "truncated": False,
        "errors": errors,
    }


def truncate_node(state: PipelineState) -> dict:
    """Truncate documents to max_context_tokens before budget pre-check and LLM calls."""
    documents = state.get("documents", [])
    if state.get("no_llm") or state.get("no_condense") or not documents:
        return {"dropped_documents": [], "truncated": False}

    selected_model = state.get("model") or os.getenv("DOC2SKILL_LLM_MODEL", DEFAULT_MODEL)
    max_context_tokens = state.get("max_context_tokens", 500_000)
    try:
        kept, dropped, _ = fit_documents_to_context(
            documents, selected_model, max_context_tokens, client=None
        )
    except ValueError as exc:
        errors = list(state.get("errors", []))
        errors.append(str(exc))
        return {"errors": errors, "skip_llm": True}

    return {
        "documents": kept,
        "distilled_sources": [doc.path for doc in kept],
        "dropped_documents": dropped,
        "truncated": bool(dropped),
    }


def budget_guard_node(state: PipelineState) -> dict:
    """Pre-check total input token count (frontmatter + condensation) before LLM calls."""
    if state.get("skip_llm") or state.get("errors"):
        return {"estimated_tokens": 0, "skip_llm": True, "llm_provider": None}

    documents = state.get("documents", [])
    max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    no_llm = state.get("no_llm", False)
    no_condense = state.get("no_condense", False)
    provider = state.get("llm_provider")
    selected_model = state.get("model") or os.getenv("DOC2SKILL_LLM_MODEL", DEFAULT_MODEL)

    if provider is None and not no_llm and os.environ.get("GEMINI_API_KEY"):
        try:
            provider = GeminiProvider(model=selected_model)
        except (RuntimeError, ValueError, OSError):
            provider = None

    frontmatter_est = estimate_prompt_tokens(documents, provider=provider)
    condense_est = 0 if no_condense else estimate_condense_tokens(documents, provider=provider)
    estimated = frontmatter_est + condense_est
    errors = list(state.get("errors", []))

    if no_llm:
        return {
            "estimated_tokens": estimated,
            "skip_llm": True,
            "llm_provider": provider,
        }

    # Pre-call budget enforcement: blocks both frontmatter & condense calls
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
    if any("Failed to load" in err for err in state.get("errors", [])):
        return "skip"
    return "distill"


def distill_node(state: PipelineState) -> dict:
    """Invoke LLM provider to synthesize frontmatter and condense the document corpus."""
    documents = state.get("documents", [])
    provider = state.get("llm_provider")
    errors = list(state.get("errors", []))
    selected_model = state.get("model") or os.getenv("DOC2SKILL_LLM_MODEL", DEFAULT_MODEL)
    no_condense = state.get("no_condense", False)

    def _fallback_result(err_msg: str | None = None) -> dict:
        if err_msg:
            errors.append(err_msg)
        return {
            "distilled": format_raw_documents_body(documents),
            "distilled_sources": [doc.path for doc in documents],
            "dropped_documents": list(state.get("dropped_documents") or []),
            "truncated": bool(state.get("truncated", False)),
            "frontmatter": None,
            "errors": errors,
        }

    if provider is None:
        try:
            provider = GeminiProvider(model=selected_model)
        except (RuntimeError, ValueError, OSError) as exc:
            return _fallback_result(f"Failed to initialize LLM provider: {exc}")

    try:
        fm_result = provider.synthesize_frontmatter(
            documents, max_tokens=state.get("max_tokens", 1000)
        )
        frontmatter = {
            "description": fm_result.description,
            "when_to_use": fm_result.when_to_use,
        }
        in_tokens = fm_result.input_tokens
        out_tokens = fm_result.output_tokens

        if no_condense:
            return {
                "distilled": format_raw_documents_body(documents),
                "distilled_sources": [doc.path for doc in documents],
                "dropped_documents": list(state.get("dropped_documents") or []),
                "truncated": bool(state.get("truncated", False)),
                "frontmatter": frontmatter,
                "condense_input_tokens": 0,
                "condense_output_tokens": 0,
                "llm_input_tokens": in_tokens,
                "llm_output_tokens": out_tokens,
                "actual_tokens": in_tokens + out_tokens,
            }

        from doc2skill.distill.condense import condense

        condense_kwargs: dict = {
            "documents": documents,
            "model": selected_model,
            "max_output_tokens": state.get("max_output_tokens", 15_000),
            "max_context_tokens": state.get("max_context_tokens", 500_000),
        }
        provider_client = getattr(provider, "client", None)
        if (
            provider_client is not None
            and "client" in inspect.signature(condense).parameters
        ):
            condense_kwargs["client"] = provider_client

        cond_result = _run_async(condense(**condense_kwargs))

        total_in = in_tokens + cond_result.input_tokens
        total_out = out_tokens + cond_result.output_tokens
        merged_dropped = list(
            dict.fromkeys(
                [*(state.get("dropped_documents") or []), *cond_result.dropped_documents]
            )
        )
        return {
            "distilled": cond_result.body,
            "distilled_sources": cond_result.sources,
            "dropped_documents": merged_dropped,
            "truncated": bool(state.get("truncated") or cond_result.truncated),
            "frontmatter": frontmatter,
            "condense_input_tokens": cond_result.input_tokens,
            "condense_output_tokens": cond_result.output_tokens,
            "llm_input_tokens": total_in,
            "llm_output_tokens": total_out,
            "actual_tokens": total_in + total_out,
        }
    except (genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
        return _fallback_result(f"LLM synthesis failed: {exc}")


def budget_record_node(state: PipelineState) -> dict:
    """Post-call accounting node (checks if actual spend exceeded max_tokens)."""
    actual = state.get("actual_tokens", 0)
    max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    warnings = list(state.get("warnings", []))

    if actual > max_tokens:
        warnings.append(
            f"Budget warning: actual {actual} tokens exceeded maximum allowed {max_tokens}."
        )

    raw_docs = state.get("documents", [])
    distilled = state.get("distilled") or format_raw_documents_body(raw_docs)
    distilled_sources = state.get("distilled_sources") or [d.path for d in raw_docs]
    return {
        "warnings": warnings,
        "distilled": distilled,
        "distilled_sources": distilled_sources,
    }


def emit_skill_node(state: PipelineState) -> dict:
    """Render and write SKILL.md output using condensed body, sources, and frontmatter."""
    source_path = state.get("source_path", "")
    raw_docs = state.get("documents", [])
    body = state.get("distilled") or format_raw_documents_body(raw_docs)
    sources = state.get("distilled_sources") or [doc.path for doc in raw_docs]
    skill_name = derive_skill_name(source_path) if source_path else "skill"
    frontmatter = state.get("frontmatter")
    fallback_desc = extract_description(raw_docs)

    resolved_name, skill_md = render(
        body=body,
        sources=sources,
        frontmatter=frontmatter,
        source_path=source_path,
        skill_name=skill_name,
        fallback_description=fallback_desc,
    )

    output_dir = state.get("output_dir", "./output")
    out_file = write_skill_file(skill_md, resolved_name, output_dir=output_dir)
    return {
        "distilled": body,
        "distilled_sources": sources,
        "skill_md": skill_md,
        "output_path": str(out_file),
    }


def build_pipeline_graph():
    """Build and compile the LangGraph pipeline:

    load_documents -> truncate -> budget_guard -> [distill ->] budget_record -> emit_skill
    """
    workflow = StateGraph(PipelineState)

    workflow.add_node("load_documents", load_documents_node)
    workflow.add_node("truncate", truncate_node)
    workflow.add_node("budget_guard", budget_guard_node)
    workflow.add_node("distill", distill_node)
    workflow.add_node("budget_record", budget_record_node)
    workflow.add_node("emit_skill", emit_skill_node)

    workflow.add_edge(START, "load_documents")
    workflow.add_edge("load_documents", "truncate")
    workflow.add_edge("truncate", "budget_guard")
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
