import re
from dataclasses import dataclass

from doc2skill.validate.allowlist import is_known_external
from doc2skill.validate.extract import Identifier


@dataclass
class ValidationReport:
    """Verification breakdown for extracted condensed-skill identifiers."""

    verified: list[Identifier]
    known_external: list[Identifier]
    unverified: list[Identifier]

    @property
    def precision(self) -> float:
        """verified / (verified + unverified). Known-external excluded."""
        denom = len(self.verified) + len(self.unverified)
        return len(self.verified) / denom if denom else 1.0


def _is_in_source(identifier: Identifier, source_corpus: str) -> bool:
    """Return True if identifier passes kind-appropriate matching against source_corpus."""
    if not source_corpus:
        return False

    if identifier.kind in ("class", "function", "method"):
        return (
            re.search(rf"\b{re.escape(identifier.text)}\b", source_corpus)
            is not None
        )

    # endpoint, error_code, env_var, or param: exact case-sensitive substring match
    return identifier.text in source_corpus


def verify_identifiers(
    identifiers: list[Identifier],
    source_corpus: str,
) -> ValidationReport:
    """Classify extracted identifiers into verified, known_external, and unverified."""
    verified: list[Identifier] = []
    known_external: list[Identifier] = []
    unverified: list[Identifier] = []

    for ident in identifiers:
        if is_known_external(ident):
            known_external.append(ident)
        elif _is_in_source(ident, source_corpus):
            verified.append(ident)
        else:
            unverified.append(ident)

    return ValidationReport(
        verified=verified,
        known_external=known_external,
        unverified=unverified,
    )

