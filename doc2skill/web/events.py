import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from doc2skill.cache import disk as cache
from doc2skill.emit.skill import derive_skill_name
from doc2skill.llm.gemini import DEFAULT_MODEL
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


def sse_event(event: str, data: dict[str, Any]) -> dict[str, str]:
    """Format an event dictionary for sse-starlette EventSourceResponse."""
    return {
        "event": event,
        "data": json.dumps(data),
    }


def _map_node_update(
    node_name: str,
    payload: dict[str, Any],
    *,
    source: str,
    source_type: str,
    max_tokens: int,
    model_name: str,
    no_llm: bool,
) -> tuple[list[dict[str, str]], str | None, bool]:
    """Map a single LangGraph node update to SSE event(s), next stage, and terminal flag.

    Returns (events, next_stage, should_stop).
    """
    if node_name == "load_documents":
        errors = payload.get("errors") or []
        docs = payload.get("documents") or []
        if errors:
            return (
                [sse_event("error", {"stage": "ingest", "message": errors[-1]})],
                None,
                True,
            )
        if not docs:
            return (
                [
                    sse_event(
                        "error",
                        {
                            "stage": "ingest",
                            "message": f"No documents loaded from {source}",
                        },
                    )
                ],
                None,
                True,
            )
        count = len(docs)
        return (
            [
                sse_event(
                    "ingest_done",
                    {
                        "documents": count,
                        "source_type": source_type,
                        "pages_total": count,
                    },
                )
            ],
            "budget",
            False,
        )

    if node_name == "budget_guard":
        est = int(payload.get("estimated_tokens", 0))
        errors = payload.get("errors") or []
        events = [
            sse_event(
                "budget_pre",
                {
                    "estimated_input_tokens": est,
                    "max_tokens": max_tokens,
                },
            )
        ]
        if any("Budget exceeded" in err for err in errors):
            events.append(
                sse_event(
                    "error",
                    {
                        "stage": "budget",
                        "message": f"Estimated {est} tokens > max {max_tokens}",
                    },
                )
            )
            return events, None, True

        if not payload.get("skip_llm", False) and not no_llm:
            events.append(sse_event("llm_call", {"model": model_name}))
            return events, "llm", False

        return events, "render", False

    if node_name == "distill":
        errors = payload.get("errors") or []
        if errors:
            return (
                [sse_event("error", {"stage": "llm", "message": errors[-1]})],
                None,
                True,
            )
        return (
            [
                sse_event(
                    "llm_done",
                    {
                        "input_tokens": int(payload.get("llm_input_tokens", 0)),
                        "output_tokens": int(payload.get("llm_output_tokens", 0)),
                        "actual_tokens": int(payload.get("actual_tokens", 0)),
                    },
                )
            ],
            "render",
            False,
        )

    if node_name == "emit_skill":
        skill_md = payload.get("skill_md") or ""
        skill_name = derive_skill_name(source)
        return (
            [
                sse_event(
                    "complete",
                    {
                        "skill_name": skill_name,
                        "skill_md": skill_md,
                        "cache_hit": False,
                    },
                )
            ],
            None,
            True,
        )

    return [], None, False


async def pipeline_event_stream(
    source: str,
    source_type: str = "url",
    max_pages: int = 100,
    max_tokens: int = 50_000,
    no_cache: bool = False,
    no_llm: bool = False,
    request: Request | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Stream LangGraph pipeline execution updates as SSE events."""
    yield sse_event("start", {"source": source, "source_type": source_type})

    selected_model = os.environ.get("DOC2SKILL_LLM_MODEL") or DEFAULT_MODEL
    key = cache.cache_key(
        source,
        max_tokens=max_tokens,
        model=selected_model,
        source_type=source_type,
    )

    cached_skill = None if no_cache else cache.get(key)
    if cached_skill is not None:
        yield sse_event(
            "complete",
            {
                "skill_name": derive_skill_name(source),
                "skill_md": cached_skill,
                "cache_hit": True,
            },
        )
        return

    yield sse_event("ingest_progress", {"status": "loading"})

    current_stage = "ingest"
    completed = False

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph = build_pipeline_graph()
        state = create_initial_state(
            source_path=source,
            max_tokens=max_tokens,
            output_dir=tmp_dir,
            no_llm=no_llm,
            model=selected_model,
            max_pages=max_pages,
        )
        model_name = str(state.get("model", selected_model))

        stream = graph.astream(state, stream_mode="updates")
        try:
            async for update in stream:
                if request is not None and await request.is_disconnected():
                    return

                node_name = next(iter(update))
                payload = update[node_name]

                events, next_stage, should_stop = _map_node_update(
                    node_name,
                    payload,
                    source=source,
                    source_type=source_type,
                    max_tokens=max_tokens,
                    model_name=model_name,
                    no_llm=no_llm,
                )
                if node_name == "emit_skill" and not no_cache:
                    skill_md = payload.get("skill_md") or ""
                    if skill_md:
                        cache.set(key, skill_md)

                for ev in events:
                    yield ev

                if next_stage:
                    current_stage = next_stage

                if should_stop:
                    completed = True
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield sse_event("error", {"stage": current_stage, "message": str(exc)})
            return
        finally:
            await stream.aclose()

    if not completed:
        yield sse_event(
            "error",
            {
                "stage": "pipeline",
                "message": "Pipeline completed without generating SKILL.md",
            },
        )

