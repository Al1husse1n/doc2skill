import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from doc2skill.ingest.base import Document
from doc2skill.llm.gemini import DEFAULT_MODEL, gemini_retry

load_dotenv()

logger = logging.getLogger(__name__)

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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Export it before running.")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

    kept_docs = list(documents)
    dropped_documents: list[str] = []
    truncated = False

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
