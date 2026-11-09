from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents an ingested documentation file."""

    path: str
    content: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
