from collections.abc import Sequence
from dataclasses import dataclass

from doc2skill.ingest.base import Document

DEFAULT_MAX_TOKENS = 50_000


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text using ~4 characters per token heuristic."""
    if not text:
        return 0
    # Minimum of 1 token for non-empty text
    return max(1, len(text) // 4)


def estimate_documents_tokens(documents: Sequence[Document]) -> int:
    """Estimate total tokens across a collection of Documents."""
    return sum(estimate_tokens(doc.content) for doc in documents)


@dataclass
class TokenBudget:
    """Tracks token budget constraints and token estimates."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    estimated_tokens: int = 0
    actual_tokens: int = 0

    def estimate(self, documents: Sequence[Document]) -> int:
        """Calculate and store estimated tokens for the provided documents."""
        self.estimated_tokens = estimate_documents_tokens(documents)
        return self.estimated_tokens

    def is_exceeded(self, tokens: int | None = None) -> bool:
        """Check if token count exceeds the maximum allowed budget."""
        check_val = self.estimated_tokens if tokens is None else tokens
        return check_val > self.max_tokens

    def check(self, tokens: int | None = None) -> str | None:
        """Return an error message if the budget is exceeded, or None."""
        check_val = self.estimated_tokens if tokens is None else tokens
        if check_val > self.max_tokens:
            return (
                f"Token budget exceeded: estimated {check_val} tokens "
                f"exceeds maximum allowed {self.max_tokens} tokens."
            )
        return None
