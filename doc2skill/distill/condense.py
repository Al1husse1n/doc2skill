import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from google import genai
from google.genai import types

from doc2skill.ingest.base import Document
from doc2skill.llm.gemini import DEFAULT_MODEL, gemini_retry

logger = logging.getLogger(__name__)

MIN_DOCS_BEFORE_REFUSAL = 5

CONDENSE_SYSTEM_PROMPT = """You are writing an agent skill. An agent skill is a compact reference that another AI agent loads to accomplish real tasks. It is NOT documentation. It is NOT marketing. It is a working manual.

Required output sections (use exact level-2 markdown headings):
## API surface
(function/class/endpoint signatures, grouped logically)

## Common workflows
(task-oriented workflows with 3-7 numbered steps each and exact code usage)

## Gotchas
(things that silently fail, subtle defaults, resource/lifecycle traps, or surprise users)

## Reference
(brief pointers only — no re-dumps of raw docs)

Strictly FORBIDDEN content (omit completely, even if present in source docs):
- Installation instructions (pip install, setup guides, dependencies)
- Contributing guidelines or developer workflow docs
- Changelog, release notes, or "What's new"
- License text or copyright notices
- Marketing copy ("blazingly fast", "enterprise-grade", promotional intros)
- Tables of contents
- "Further reading" or "External links" sections

Rules:
1. Every code identifier (function name, class name, parameter name, endpoint path, HTTP method, error code, environment variable) MUST appear verbatim as it does in the source docs. Do NOT paraphrase POST /v1/charges as "the charge creation endpoint".
2. Be conservative: If you are not sure a capability exists, do not include it. Do not infer from similar tools.
3. Aim for 4,000-10,000 tokens of output. A skill longer than 15,000 tokens is too long for an agent to use comfortably.
4. Respond directly with clean GitHub-flavored Markdown containing the required sections and no outer code-fence wrapper."""


@dataclass
class CondensedResult:
    """Result of single-call corpus condensation."""

    body: str
    sources: list[str]
    input_tokens: int
    output_tokens: int
    dropped_documents: list[str]
    truncated: bool


def _strip_outer_code_fences(text: str) -> str:
    """Strip accidental leading/trailing markdown code fences around the entire response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _doc_depth_key(doc: Document) -> tuple[int, str]:
    """Sort key (URL/path depth, alphabetical path) keeping shallow core docs first."""
    raw = doc.path.strip()
    if raw.startswith(("http://", "https://")):
        path_part = urlparse(raw).path.strip("/")
    else:
        path_part = raw.replace("\\", "/").strip("/")
    depth = len([seg for seg in path_part.split("/") if seg]) if path_part else 0
    return (depth, raw)


def build_condense_prompt(
    documents: Sequence[Document],
    dropped_documents: Sequence[str] = (),
) -> str:
    """Concatenate corpus documents with explicit delimiters and truncation notice."""
    header_lines = [f"Total documents in corpus: {len(documents)}"]
    if dropped_documents:
        dropped_str = ", ".join(dropped_documents)
        header_lines.append(
            f"Dropped {len(dropped_documents)} documents due to context limit: {dropped_str}"
        )

    blocks: list[str] = []
    for doc in documents:
        blocks.append(
            f"=== DOCUMENT: {doc.path} ===\n{doc.content.strip()}\n=== END DOCUMENT ==="
        )

    return "\n\n".join(["\n".join(header_lines), *blocks])


@gemini_retry
def _call_count_tokens(client: Any, model: str, contents: str) -> int:
    """Count tokens using Gemini SDK count_tokens with length-heuristic fallback."""
    try:
        resp = client.models.count_tokens(model=model, contents=contents)
        total = getattr(resp, "total_tokens", None)
        if total is not None and not isinstance(total, (bool,)):
            return int(total)
    except (OSError, RuntimeError, ValueError, AttributeError, TypeError):
        pass
    return max(1, len(contents) // 4)


@gemini_retry
def _call_generate_condense(
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> Any:
    """Invoke Gemini generate_content for corpus condensation with retry policy."""
    config = types.GenerateContentConfig(
        system_instruction=CONDENSE_SYSTEM_PROMPT,
        response_mime_type="text/plain",
        max_output_tokens=max_output_tokens,
    )
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )


def _create_client() -> Any:
    """Instantiate default Gemini client using GEMINI_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Export it before running.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )


def fit_documents_to_context(
    documents: Sequence[Document],
    model: str,
    max_context_tokens: int,
    client: Any | None = None,
) -> tuple[list[Document], list[str], int]:
    """Fit documents under max_context_tokens by dropping deeply-nested/tail docs.

    Returns (kept_docs, dropped_paths, token_count).
    Raises ValueError if even 5 docs (or fewer) still exceed max_context_tokens.
    """
    kept = list(documents)
    prompt = build_condense_prompt(kept)
    if client is not None:
        total = _call_count_tokens(client, model, prompt)
    else:
        total = max(1, len(prompt) // 4)

    if total <= max_context_tokens:
        return kept, [], total

    initial_total = total
    min_keep = min(MIN_DOCS_BEFORE_REFUSAL, len(kept))
    sorted_docs = sorted(kept, key=_doc_depth_key)
    dropped: list[str] = []

    while total > max_context_tokens and len(sorted_docs) > min_keep:
        # Estimate how many tail docs to drop to cover excess without excessive count_tokens calls
        excess = total - max_context_tokens
        reclaimed = 0
        while reclaimed < excess and len(sorted_docs) > min_keep:
            tail_doc = sorted_docs.pop()
            dropped.append(tail_doc.path)
            reclaimed += max(1, len(tail_doc.content) // 4)

        prompt = build_condense_prompt(sorted_docs, dropped_documents=dropped)
        if client is not None:
            total = _call_count_tokens(client, model, prompt)
        else:
            total = max(1, len(prompt) // 4)

    if total > max_context_tokens:
        raise ValueError("Docs exceed context limit even after truncation")

    if dropped:
        logger.warning(
            "Dropped %d documents to fit context (%d > %d)",
            len(dropped),
            initial_total,
            max_context_tokens,
        )

    return sorted_docs, dropped, total


async def condense(
    documents: Sequence[Document],
    model: str | None = None,
    max_output_tokens: int = 15_000,
    max_context_tokens: int = 500_000,
    client: Any | None = None,
) -> CondensedResult:
    """Condense all ingested documents into a single compact agent skill body."""
    if not documents:
        raise ValueError("Cannot condense empty document list.")

    selected_model = model or os.environ.get("DOC2SKILL_LLM_MODEL") or DEFAULT_MODEL
    if client is None:
        client = _create_client()

    kept_docs, dropped_documents, _ = await asyncio.to_thread(
        fit_documents_to_context,
        documents,
        selected_model,
        max_context_tokens,
        client,
    )
    truncated = len(dropped_documents) > 0

    user_prompt = build_condense_prompt(kept_docs, dropped_documents=dropped_documents)

    response = await asyncio.to_thread(
        _call_generate_condense,
        client,
        selected_model,
        user_prompt,
        max_output_tokens,
    )

    body = _strip_outer_code_fences(response.text or "")

    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

    raw_doc_tokens = sum(len(d.content) for d in kept_docs) // 4
    effective_in_tokens = input_tokens if input_tokens > 0 else raw_doc_tokens
    effective_out_tokens = output_tokens if output_tokens > 0 else (len(body) // 4)
    min_required = 500 if effective_in_tokens >= 400 else 50

    if not body or effective_out_tokens < min_required:
        raise ValueError("Condensation produced suspiciously short output; refusing.")

    return CondensedResult(
        body=body,
        sources=[doc.path for doc in kept_docs],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_documents=dropped_documents,
        truncated=truncated,
    )
