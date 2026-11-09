import json
import os
import re
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult

DEFAULT_MODEL = "gemini-2.5-flash"
DOC_SNIPPET_MAX_CHARS = 500


class FrontmatterSchema(BaseModel):
    """Structured JSON schema for Gemini frontmatter output."""

    description: str = Field(
        ...,
        description="One sentence, max 200 characters, summarizing what the skill does.",
    )
    when_to_use: str = Field(
        ...,
        description="2-5 markdown bullet points, each on its own line separated by newline '\\n', starting with '- '.",
    )


class GeminiProvider:
    """Gemini-backed LLM provider for frontmatter synthesis using google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("DOC2SKILL_LLM_MODEL") or DEFAULT_MODEL

        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY not set. Export it before running.")
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )

    def build_prompt(self, documents: Sequence[Document]) -> str:
        """Build single prompt with titles and first ~500 chars of each document."""
        doc_summaries: list[str] = []
        for doc in documents:
            snippet = doc.content[:DOC_SNIPPET_MAX_CHARS].strip()
            doc_summaries.append(
                f"### Document: {doc.title} (path: {doc.path})\n{snippet}"
            )

        combined_docs = "\n\n".join(doc_summaries)

        return f"""You are an expert at creating agent skills from documentation.
Analyze the following documentation snippets and generate metadata for a SKILL.md specification.

Requirements:
1. "description": Maximum 200 characters. Exactly one sentence stating what the skill does.
2. "when_to_use": 2-5 bullet points formatted with markdown '-' bullets, each starting with a trigger scenario or user intent. Explicit instruction: Do not invent capabilities not present in the docs.

Return a JSON object with exactly two string fields:
{{"description": "...", "when_to_use": "..."}}

Documentation snippets:
{combined_docs}
"""

    def count_tokens(self, prompt: str) -> int:
        """Pre-call token count using client.models.count_tokens with fallback."""
        try:
            resp = self.client.models.count_tokens(
                model=self.model,
                contents=prompt,
            )
            if hasattr(resp, "total_tokens") and resp.total_tokens is not None:
                return int(resp.total_tokens)
        except (OSError, RuntimeError, ValueError, AttributeError):
            # Fall back to length heuristic if network count fails
            return max(1, len(prompt) // 4)
        return max(1, len(prompt) // 4)

    def synthesize_frontmatter(
        self,
        documents: Sequence[Document],
        max_tokens: int = 1000,
    ) -> FrontmatterResult:
        """Synthesize description and 'When to use' section using Gemini structured output."""
        prompt = self.build_prompt(documents)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FrontmatterSchema,
            max_output_tokens=max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        raw_text = response.text or "{}"
        try:
            parsed = json.loads(raw_text)
            description = str(parsed.get("description", "")).strip()
            when_to_use = str(parsed.get("when_to_use", "")).strip()
            # Ensure each bullet point is separated by a newline
            when_to_use = re.sub(r"([^\n])\s*-\s*", r"\1\n- ", when_to_use)
        except json.JSONDecodeError:
            description = ""
            when_to_use = raw_text.strip()

        # Extract usage metadata
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return FrontmatterResult(
            description=description,
            when_to_use=when_to_use,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
