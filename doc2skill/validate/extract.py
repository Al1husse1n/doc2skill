import re
from dataclasses import dataclass

from doc2skill.validate.allowlist import KNOWN_THIRD_PARTY, STDLIB_MODULES

ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) /[\w\-/{}]+")
QUALIFIED_CLASS_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\.([A-Z]\w+)\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\.([a-z_]\w*)\(")
BARE_CLASS_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]*(?:Client|Error|Exception|Transport|Auth|Timeout|Response|Request|URL|Headers|Cookies)\b"
)
ERROR_CODE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,4}\b")
ENV_VAR_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([a-z_][a-z0-9_]*)\b|import\s+([a-z_][a-z0-9_]*(?:\s+as\s+([a-z_][a-z0-9_]*))?))",
    re.MULTILINE,
)

# Common variable/instance names used as method receivers in docs examples
_INSTANCE_RECEIVERS = {
    "app",
    "auth",
    "body",
    "buf",
    "cache",
    "cfg",
    "client",
    "cls",
    "config",
    "conn",
    "content",
    "cookies",
    "ctx",
    "cursor",
    "data",
    "db",
    "err",
    "exc",
    "extensions",
    "file",
    "fp",
    "headers",
    "history",
    "item",
    "limits",
    "logger",
    "next_request",
    "obj",
    "pool",
    "req",
    "request",
    "res",
    "resp",
    "response",
    "router",
    "self",
    "session",
    "stream",
    "text",
    "timeout",
    "transport",
    "url",
}


@dataclass
class Identifier:
    """Extracted technical identifier and surrounding context from condensed markdown."""

    text: str
    kind: str  # "function" | "class" | "method" | "endpoint" | "error_code" | "env_var" | "param"
    context: str


def _make_context(markdown: str, start: int, end: int) -> str:
    """Build a normalized 80-character context window centered around [start, end)."""
    mid = (start + end) // 2
    win_start = min(start, max(0, mid - 40))
    win_end = max(end, min(len(markdown), mid + 40))
    return " ".join(markdown[win_start:win_end].split())


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    """Return True if [start, end) overlaps any previously claimed span."""
    s, e = span
    return any(s < ce and cs < e for cs, ce in claimed)


def _collect_regex(
    pattern: re.Pattern[str],
    markdown: str,
    kind: str,
    claimed: list[tuple[int, int]],
    hits: list[tuple[int, Identifier]],
    predicate=None,
) -> None:
    """Scan non-overlapping matches for a pattern and record extracted Identifier hits."""
    for match in pattern.finditer(markdown):
        token = match.group(0)
        if predicate is not None and not predicate(token):
            continue
        span = match.span()
        if not _overlaps(span, claimed):
            claimed.append(span)
            hits.append(
                (
                    span[0],
                    Identifier(
                        text=token,
                        kind=kind,
                        context=_make_context(markdown, *span),
                    ),
                )
            )


def _discover_module_names(markdown: str) -> set[str]:
    """Collect imported module names and module prefixes from module.Class references."""
    modules = set(STDLIB_MODULES) | set(KNOWN_THIRD_PARTY)
    for match in IMPORT_RE.finditer(markdown):
        from_pkg, imp_pkg, alias = match.groups()
        if from_pkg:
            modules.add(from_pkg)
        if alias:
            modules.add(alias)
        elif imp_pkg:
            modules.add(imp_pkg)
    for match in QUALIFIED_CLASS_RE.finditer(markdown):
        modules.add(match.group(1))
    return modules


def _is_module_receiver(receiver: str, known_modules: set[str]) -> bool:
    """Distinguish module.function() from instance.method() receivers."""
    if receiver[0].isupper():
        return False
    if receiver in known_modules:
        return True
    if receiver in _INSTANCE_RECEIVERS or "_" in receiver or len(receiver) == 1:
        return False
    return True


def extract_identifiers(markdown: str) -> list[Identifier]:
    """Extract and deduplicate API/environment identifiers from markdown in first-appearance order."""
    if not markdown:
        return []

    claimed: list[tuple[int, int]] = []
    hits: list[tuple[int, Identifier]] = []
    known_modules = _discover_module_names(markdown)

    # 1. Endpoint (highest specificity)
    _collect_regex(ENDPOINT_RE, markdown, "endpoint", claimed, hits)

    # 2a. Qualified class: module.Class
    _collect_regex(QUALIFIED_CLASS_RE, markdown, "class", claimed, hits)

    # 2b + 3. Qualified function module.fn() vs instance method receiver.method()
    for match in CALL_RE.finditer(markdown):
        receiver, name = match.group(1), match.group(2)
        if _is_module_receiver(receiver, known_modules):
            span = (match.start(), match.end() - 1)  # exclude trailing "("
            if not _overlaps(span, claimed):
                claimed.append(span)
                hits.append(
                    (
                        span[0],
                        Identifier(
                            text=f"{receiver}.{name}",
                            kind="function",
                            context=_make_context(markdown, *span),
                        ),
                    )
                )
        else:
            method_span = match.span(2)
            if not _overlaps(method_span, claimed):
                claimed.append(method_span)
                hits.append(
                    (
                        method_span[0],
                        Identifier(
                            text=name,
                            kind="method",
                            context=_make_context(markdown, match.start(), match.end()),
                        ),
                    )
                )

    # 4. Bare class names ending with known API suffixes
    _collect_regex(BARE_CLASS_RE, markdown, "class", claimed, hits)

    # 5. Error codes / snake_case tokens
    _collect_regex(ERROR_CODE_RE, markdown, "error_code", claimed, hits)

    # 6. Environment variables (contains '_' OR >= 6 uppercase chars)
    _collect_regex(
        ENV_VAR_RE,
        markdown,
        "env_var",
        claimed,
        hits,
        predicate=lambda tok: "_" in tok or len(tok) >= 6,
    )

    # Preserve order of first appearance in source markdown and deduplicate by (text, kind)
    hits.sort(key=lambda item: item[0])
    seen: set[tuple[str, str]] = set()
    result: list[Identifier] = []
    for _, ident in hits:
        key = (ident.text, ident.kind)
        if key not in seen:
            seen.add(key)
            result.append(ident)

    return result

