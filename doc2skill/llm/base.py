from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from doc2skill.ingest.base import Document


@dataclass
class FrontmatterResult:
    """Result of LLM frontmatter synthesis."""

    description: str
    when_to_use: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    """Protocol defining the interface for frontmatter synthesis providers."""

    def synthesize_frontmatter(
        self,
        documents: Sequence[Document],
        max_tokens: int,
    ) -> FrontmatterResult:
        """Synthesize description and 'When to use' section from ingested documents."""
        ...
