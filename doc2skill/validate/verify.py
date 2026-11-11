import re
from dataclasses import dataclass, replace

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


def _verify_dotted_class(identifier: Identifier, source_corpus: str) -> str | None:
    """Verify dotted class name via exact match or bare heading/class/module fallback.

    Returns verification method ('full' or 'bare'), or None if unverified.
    """
    if re.search(rf"\b{re.escape(identifier.text)}\b", source_corpus):
        return "full"

    bare_name = identifier.text.split(".")[-1]
    if not re.search(rf"\b{re.escape(bare_name)}\b", source_corpus):
        return None

    escaped_bare = re.escape(bare_name)
    heading_re = re.compile(rf"^#{{1,6}}\s+`?{escaped_bare}`?\s*$", re.MULTILINE)
    class_def_re = re.compile(rf"\bclass\s+{escaped_bare}\b")
    module_ref_re = re.compile(
        rf"(?:\b\w+\.{escaped_bare}\b|\bimport\b[^\n]*\b{escaped_bare}\b)"
    )

    if (
        heading_re.search(source_corpus)
        or class_def_re.search(source_corpus)
        or module_ref_re.search(source_corpus)
    ):
        return "bare"

    return None


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
        elif ident.kind == "class" and "." in ident.text:
            method = _verify_dotted_class(ident, source_corpus) if source_corpus else None
            if method is not None:
                verified.append(replace(ident, context=method))
            else:
                unverified.append(ident)
        elif _is_in_source(ident, source_corpus):
            verified.append(ident)
        else:
            unverified.append(ident)

    return ValidationReport(
        verified=verified,
        known_external=known_external,
        unverified=unverified,
    )
