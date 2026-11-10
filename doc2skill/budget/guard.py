from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from doc2skill.ingest.base import Document

DEFAULT_MAX_TOKENS = 50_000


def fallback_estimate_tokens(text: str) -> int:
    """Heuristic token estimate (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_prompt_tokens(
    documents: Sequence[Document],
    provider: Any | None = None,
) -> int:
    """Pre-call estimate using provider count_tokens or fallback heuristic."""
    if not documents:
        return 0

    if (
        provider is not None
        and hasattr(provider, "build_prompt")
        and hasattr(provider, "count_tokens")
    ):
        try:
            prompt = provider.build_prompt(documents)
            return provider.count_tokens(prompt)
        except (OSError, RuntimeError, ValueError, AttributeError):
            pass

    # Fallback heuristic across document contents
    total_chars = sum(len(doc.content[:500]) for doc in documents)
    return max(1, total_chars // 4)


def estimate_condense_tokens(
    documents: Sequence[Document],
    provider: Any | None = None,
) -> int:
    """Pre-call estimate for full-corpus condensation input prompt."""
    if not documents:
        return 0

    from doc2skill.distill.condense import build_condense_prompt

    prompt = build_condense_prompt(documents)
    if provider is not None and hasattr(provider, "count_tokens"):
        try:
            return provider.count_tokens(prompt)
        except (OSError, RuntimeError, ValueError, AttributeError):
            pass

    return max(1, len(prompt) // 4)


def estimate_tokens(text: str) -> int:
    """Public helper for estimated token count."""
    return fallback_estimate_tokens(text)


def estimate_documents_tokens(documents: Sequence[Document]) -> int:
    """Public helper for document token estimation."""
    return sum(fallback_estimate_tokens(doc.content) for doc in documents)


@dataclass
class TokenBudget:
    """Tracks token budget constraints and token estimates."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    estimated_tokens: int = 0
    actual_tokens: int = 0

    def estimate(
        self,
        documents: Sequence[Document],
        provider: Any | None = None,
    ) -> int:
        """Calculate and store estimated tokens for the provided documents."""
        self.estimated_tokens = estimate_prompt_tokens(documents, provider=provider)
        return self.estimated_tokens

    def is_exceeded(self, tokens: int | None = None) -> bool:
        """Check if token count exceeds the maximum allowed budget."""
        check_val = self.estimated_tokens if tokens is None else tokens
        return check_val > self.max_tokens

    def check(self, tokens: int | None = None) -> str | None:
        """Return error message if budget is exceeded before LLM spend."""
        check_val = self.estimated_tokens if tokens is None else tokens
        if check_val > self.max_tokens:
            return f"Budget exceeded: estimated {check_val} input tokens > max {self.max_tokens}. LLM call skipped."
        return None
