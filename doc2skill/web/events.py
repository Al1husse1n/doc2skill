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


def _extract_sections(markdown_body: str) -> list[str]:
    """Extract level-2 (## ...) markdown section headers in appearance order."""
    return [
        line[3:].strip()
        for line in markdown_body.splitlines()
        if line.startswith("## ") and line[3:].strip()
    ]


def _map_node_update(
    node_name: str,
    payload: dict[str, Any],
    *,
    source: str,
    source_type: str,
    max_tokens: int,
    max_context_tokens: int,
    model_name: str,
    no_llm: bool,
    no_condense: bool,
    run_meta: dict[str, Any],
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
        events = [
            sse_event(
                "ingest_done",
                {
                    "documents": count,
                    "source_type": source_type,
                    "pages_total": count,
                },
            )
        ]
        return events, "truncate", False

    if node_name == "truncate":
        errors = payload.get("errors") or []
        if errors:
            return (
                [sse_event("error", {"stage": "truncate", "message": errors[-1]})],
                None,
                True,
            )

        events: list[dict[str, str]] = []
        dropped = list(payload.get("dropped_documents") or [])
        is_truncated = bool(payload.get("truncated", False))
        if is_truncated or dropped:
            run_meta["truncate_warning_emitted"] = True
            run_meta["truncated"] = True
            run_meta["dropped_documents"] = dropped
            events.append(
                sse_event(
                    "truncate_warning",
                    {
                        "dropped_count": len(dropped),
                        "reason": "context_limit",
                    },
                )
            )

        return events, "budget", False

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

        events: list[dict[str, str]] = []
        is_truncated = bool(payload.get("truncated", False))
        dropped = list(payload.get("dropped_documents") or [])
        if is_truncated:
            run_meta["truncated"] = True
            if dropped:
                run_meta["dropped_documents"] = dropped
            if not run_meta.get("truncate_warning_emitted"):
                run_meta["truncate_warning_emitted"] = True
                events.append(
                    sse_event(
                        "truncate_warning",
                        {
                            "dropped_count": len(dropped),
                            "reason": "context_limit",
                        },
                    )
                )

        events.append(
            sse_event(
                "llm_done",
                {
                    "input_tokens": int(payload.get("llm_input_tokens", 0)),
                    "output_tokens": int(payload.get("llm_output_tokens", 0)),
                    "actual_tokens": int(payload.get("actual_tokens", 0)),
                },
            )
        )

        if not no_condense:
            cond_in = int(payload.get("condense_input_tokens", 0))
            cond_out = int(payload.get("condense_output_tokens", 0))
            body = str(payload.get("distilled") or "")
            events.append(
                sse_event(
                    "condensed",
                    {
                        "input_tokens": cond_in,
                        "output_tokens": cond_out,
                        "sections": _extract_sections(body),
                    },
                )
            )

        return events, "render", False

    if node_name == "validate":
        errors = payload.get("errors") or []
        report = payload.get("validation_report")
        if report is None:
            if errors:
                return (
                    [sse_event("error", {"stage": "validate", "message": errors[-1]})],
                    None,
                    True,
                )
            return [], "render", False

        verified = list(report.get("verified") or [])
        known_external = list(report.get("known_external") or [])
        unverified = list(report.get("unverified") or [])
        denom = len(verified) + len(unverified)
        raw_precision = (
            float(report["precision"])
            if "precision" in report
            else (len(verified) / denom if denom else 1.0)
        )
        precision = round(raw_precision, 3)

        run_meta["validation"] = {
            "verified_count": len(verified),
            "known_external_count": len(known_external),
            "unverified_count": len(unverified),
            "precision": precision,
            "unverified": [
                {
                    "text": str(item.get("text", "")),
                    "kind": str(item.get("kind", "")),
                    "context": str(item.get("context", "")),
                }
                for item in unverified
            ],
        }

        events = [
            sse_event(
                "validation",
                {
                    "verified_count": len(verified),
                    "known_external_count": len(known_external),
                    "unverified_count": len(unverified),
                    "precision": precision,
                    "unverified_preview": [
                        str(item.get("text", "")) for item in unverified[:5]
                    ],
                },
            )
        ]

        if errors:
            events.append(
                sse_event("error", {"stage": "validate", "message": errors[-1]})
            )
            return events, None, True

        return events, "render", False

    if node_name == "emit_skill":
        skill_md = payload.get("skill_md") or ""
        skill_name = derive_skill_name(source)
        is_truncated = bool(run_meta.get("truncated", False))
        complete_data: dict[str, Any] = {
            "skill_name": skill_name,
            "skill_md": skill_md,
            "cache_hit": False,
            "truncated": is_truncated,
        }
        if is_truncated:
            complete_data["dropped_documents"] = list(
                run_meta.get("dropped_documents") or []
            )
        if run_meta.get("validation") is not None:
            complete_data["validation"] = run_meta["validation"]
        return (
            [sse_event("complete", complete_data)],
            None,
            True,
        )

    return [], None, False


async def pipeline_event_stream(
    source: str,
    source_type: str = "url",
    max_pages: int = 100,
    max_tokens: int = 50_000,
    max_output_tokens: int = 15_000,
    max_context_tokens: int = 500_000,
    no_cache: bool = False,
    no_llm: bool = False,
    no_condense: bool = False,
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

    bypass_cache = no_cache or no_condense
    cached_skill = None if bypass_cache else cache.get(key)
    if cached_skill is not None:
        yield sse_event(
            "complete",
            {
                "skill_name": derive_skill_name(source),
                "skill_md": cached_skill,
                "cache_hit": True,
                "truncated": False,
            },
        )
        return

    yield sse_event("ingest_progress", {"status": "loading"})

    current_stage = "ingest"
    completed = False
    run_meta: dict[str, Any] = {
        "truncated": False,
        "dropped_documents": [],
        "truncate_warning_emitted": False,
        "validation": None,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph = build_pipeline_graph()
        state = create_initial_state(
            source_path=source,
            max_tokens=max_tokens,
            output_dir=tmp_dir,
            no_llm=no_llm,
            model=selected_model,
            max_pages=max_pages,
            max_output_tokens=max_output_tokens,
            max_context_tokens=max_context_tokens,
            no_condense=no_condense,
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
                    max_context_tokens=max_context_tokens,
                    model_name=model_name,
                    no_llm=no_llm,
                    no_condense=no_condense,
                    run_meta=run_meta,
                )
                if node_name == "emit_skill" and not bypass_cache:
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
