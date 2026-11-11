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


@dataclass
class _CodeScanState:
    """Per-character lexical metadata for filtering example-scoped code tokens."""

    in_string: list[bool]
    paren_depth: list[int]
    in_brace_top: list[bool]
    class_body_lines: set[int]


def _make_context(text: str, start: int, end: int) -> str:
    """Build a normalized 80-character context window centered around [start, end)."""
    mid = (start + end) // 2
    win_start = min(start, max(0, mid - 40))
    win_end = max(end, min(len(text), mid + 40))
    return " ".join(text[win_start:win_end].split())


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    """Return True if [start, end) overlaps any previously claimed span."""
    s, e = span
    return any(s < ce and cs < e for cs, ce in claimed)


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


def _split_markdown(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into non-fenced prose text and a list of (fence_body, language)."""
    prose_lines: list[str] = []
    code_blocks: list[tuple[str, str]] = []
    in_fence = False
    fence_lang = ""
    fence_lines: list[str] = []

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = stripped[3:].strip()
                fence_lines = []
            else:
                in_fence = False
                code_blocks.append(("\n".join(fence_lines), fence_lang))
                prose_lines.append("")
                fence_lang = ""
                fence_lines = []
            continue

        if in_fence:
            fence_lines.append(line)
        else:
            prose_lines.append(line)

    if in_fence and fence_lines:
        code_blocks.append(("\n".join(fence_lines), fence_lang))

    return "\n".join(prose_lines), code_blocks


def _scan_code_state(code: str) -> _CodeScanState:
    """Scan a code block once to track strings, bracket nesting, and class-body lines."""
    n = len(code)
    in_string = [False] * n
    paren_depth = [0] * n
    in_brace_top = [False] * n

    stack: list[str] = []
    quote: str | None = None
    in_comment = False
    i = 0

    while i < n:
        ch = code[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
            paren_depth[i] = len(stack)
            in_brace_top[i] = bool(stack and stack[-1] == "{")
            i += 1
            continue

        if quote is not None:
            in_string[i] = True
            paren_depth[i] = len(stack)
            in_brace_top[i] = bool(stack and stack[-1] == "{")
            if ch == "\\" and len(quote) == 1 and i + 1 < n:
                i += 1
                in_string[i] = True
                paren_depth[i] = len(stack)
                in_brace_top[i] = bool(stack and stack[-1] == "{")
                i += 1
                continue
            if code.startswith(quote, i):
                for j in range(1, len(quote)):
                    if i + j < n:
                        in_string[i + j] = True
                        paren_depth[i + j] = len(stack)
                        in_brace_top[i + j] = bool(stack and stack[-1] == "{")
                i += len(quote)
                quote = None
                continue
            i += 1
            continue

        if ch == "#":
            in_comment = True
            paren_depth[i] = len(stack)
            in_brace_top[i] = bool(stack and stack[-1] == "{")
            i += 1
            continue

        if code.startswith('"""', i) or code.startswith("'''", i):
            quote = code[i : i + 3]
            for j in range(3):
                if i + j < n:
                    in_string[i + j] = True
                    paren_depth[i + j] = len(stack)
                    in_brace_top[i + j] = bool(stack and stack[-1] == "{")
            i += 3
            continue

        if ch in ('"', "'"):
            quote = ch
            in_string[i] = True
            paren_depth[i] = len(stack)
            in_brace_top[i] = bool(stack and stack[-1] == "{")
            i += 1
            continue

        paren_depth[i] = len(stack)
        in_brace_top[i] = bool(stack and stack[-1] == "{")

        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}" and stack:
            expected = {"(": ")", "[": "]", "{": "}"}[stack[-1]]
            if ch == expected:
                stack.pop()

        i += 1

    # Identify line numbers at statement level inside a `class ...:` body
    class_body_lines: set[int] = set()
    active_class_indents: list[int] = []
    offset = 0
    for line_idx, line in enumerate(code.splitlines(keepends=True)):
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)
        if stripped and not stripped.startswith("#"):
            depth_at_line = paren_depth[offset + indent] if offset + indent < n else 0
            if depth_at_line == 0:
                while active_class_indents and indent <= active_class_indents[-1]:
                    active_class_indents.pop()
                if active_class_indents:
                    class_body_lines.add(line_idx)
                if re.match(r"class\s+[A-Za-z_]", stripped):
                    active_class_indents.append(indent)
        offset += len(line)

    return _CodeScanState(
        in_string=in_string,
        paren_depth=paren_depth,
        in_brace_top=in_brace_top,
        class_body_lines=class_body_lines,
    )


def _should_skip_code_error_code(
    code: str,
    match: re.Match[str],
    state: _CodeScanState,
) -> bool:
    """Filter statement LHS assignments, string literals, def/class names, and dict keys."""
    s, e = match.span()
    if state.in_string[s]:
        return True

    prefix_short = code[max(0, s - 16) : s]
    if re.search(r"\b(?:def|class)\s+$", prefix_short):
        return True

    line_start = code.rfind("\n", 0, s) + 1
    if (
        state.paren_depth[s] == 0
        and code[line_start:s].strip() == ""
        and re.match(r"[ \t]*=(?!=)", code[e:])
    ):
        return True

    if state.in_brace_top[s] and re.match(r"""(?:['"])?[ \t]*:""", code[e:]):
        return True

    return False


def _should_skip_code_class(
    code: str,
    match: re.Match[str],
    state: _CodeScanState,
) -> bool:
    """Skip class definitions (`class Foo:`) and class-body attribute type annotations (`x: Foo`)."""
    s, _ = match.span()
    if re.search(r"\bclass\s+$", code[max(0, s - 16) : s]):
        return True

    line_idx = code.count("\n", 0, s)
    line_start = code.rfind("\n", 0, s) + 1
    if (
        line_idx in state.class_body_lines
        and state.paren_depth[s] == 0
        and re.search(r":\s*$", code[line_start:s])
    ):
        return True

    return False


def _collect_regex(
    pattern: re.Pattern[str],
    text: str,
    kind: str,
    claimed: list[tuple[int, int]],
    hits: list[tuple[int, Identifier]],
    predicate=None,
) -> None:
    """Scan non-overlapping matches for a pattern and record extracted Identifier hits."""
    for match in pattern.finditer(text):
        if predicate is not None and not predicate(match):
            continue
        span = match.span()
        if not _overlaps(span, claimed):
            claimed.append(span)
            hits.append(
                (
                    span[0],
                    Identifier(
                        text=match.group(0),
                        kind=kind,
                        context=_make_context(text, *span),
                    ),
                )
            )


def _extract_segment(
    text: str,
    known_modules: set[str],
    code_state: _CodeScanState | None = None,
) -> list[Identifier]:
    """Extract identifiers from a prose segment or fenced code body."""
    if not text:
        return []

    claimed: list[tuple[int, int]] = []
    hits: list[tuple[int, Identifier]] = []

    def class_predicate(m: re.Match[str]) -> bool:
        return True if code_state is None else not _should_skip_code_class(text, m, code_state)

    def error_code_predicate(m: re.Match[str]) -> bool:
        return (
            True
            if code_state is None
            else not _should_skip_code_error_code(text, m, code_state)
        )

    # 1. Endpoint (highest specificity)
    _collect_regex(ENDPOINT_RE, text, "endpoint", claimed, hits)

    # 2a. Qualified class: module.Class
    _collect_regex(
        QUALIFIED_CLASS_RE,
        text,
        "class",
        claimed,
        hits,
        predicate=class_predicate,
    )

    # 2b + 3. Qualified function module.fn() vs instance method receiver.method()
    for match in CALL_RE.finditer(text):
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
                            context=_make_context(text, *span),
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
                            context=_make_context(text, match.start(), match.end()),
                        ),
                    )
                )

    # 4. Bare class names ending with known API suffixes
    _collect_regex(
        BARE_CLASS_RE,
        text,
        "class",
        claimed,
        hits,
        predicate=class_predicate,
    )

    # 5. Error codes / snake_case tokens
    _collect_regex(
        ERROR_CODE_RE,
        text,
        "error_code",
        claimed,
        hits,
        predicate=error_code_predicate,
    )

    # 6. Environment variables (contains '_' OR >= 6 uppercase chars)
    _collect_regex(
        ENV_VAR_RE,
        text,
        "env_var",
        claimed,
        hits,
        predicate=lambda m: "_" in m.group(0) or len(m.group(0)) >= 6,
    )

    hits.sort(key=lambda item: item[0])
    return [ident for _, ident in hits]


def _extract_from_prose(
    prose: str, known_modules: set[str] | None = None
) -> list[Identifier]:
    """Extract all identifier patterns from non-fenced prose."""
    modules = known_modules if known_modules is not None else _discover_module_names(prose)
    return _extract_segment(prose, modules, code_state=None)


def _extract_from_code(
    code_blocks: list[tuple[str, str]],
    known_modules: set[str] | None = None,
) -> list[Identifier]:
    """Extract identifiers from fenced code blocks while skipping example-scoped tokens."""
    out: list[Identifier] = []
    for body, _lang in code_blocks:
        if not body:
            continue
        modules = known_modules if known_modules is not None else _discover_module_names(body)
        state = _scan_code_state(body)
        out.extend(_extract_segment(body, modules, code_state=state))
    return out


def _dedupe(identifiers: list[Identifier]) -> list[Identifier]:
    """Deduplicate identifiers by (text, kind) while preserving first appearance."""
    seen: set[tuple[str, str]] = set()
    result: list[Identifier] = []
    for ident in identifiers:
        key = (ident.text, ident.kind)
        if key not in seen:
            seen.add(key)
            result.append(ident)
    return result


def extract_identifiers(markdown: str) -> list[Identifier]:
    """Extract and deduplicate API/environment identifiers from markdown prose and code blocks."""
    if not markdown:
        return []

    known_modules = _discover_module_names(markdown)
    prose, code_blocks = _split_markdown(markdown)
    identifiers: list[Identifier] = []
    identifiers.extend(_extract_from_prose(prose, known_modules=known_modules))
    identifiers.extend(_extract_from_code(code_blocks, known_modules=known_modules))
    return _dedupe(identifiers)
