from doc2skill.validate.allowlist import is_known_external
from doc2skill.validate.extract import Identifier, extract_identifiers


def test_extract_all_identifier_kinds():
    """Fixture containing every identifier kind extracts expected (text, kind) pairs in order."""
    markdown = """
## API surface

Use `httpx.Client` or `AsyncClient` with `BasicAuth` for authenticated requests.
Call `httpx.get("https://api.example.com")` or `asyncio.run(main())`.

## Common workflows

1. Send `POST /v1/charges` and `GET /v1/customers/{id}` (or `DELETE /v1/sessions/{id}`).
2. Parse output with `response.json()` and guard with `response.raise_for_status()`.
3. Enable `follow_redirects=True` and handle `card_declined` / `rate_limit_exceeded`.
4. Configure `GEMINI_API_KEY`, `HTTP_PROXY`, or `SECRET` via environment variables.
"""
    extracted = extract_identifiers(markdown)
    pairs = [(item.text, item.kind) for item in extracted]

    expected_pairs = [
        ("POST /v1/charges", "endpoint"),
        ("GET /v1/customers/{id}", "endpoint"),
        ("DELETE /v1/sessions/{id}", "endpoint"),
        ("httpx.Client", "class"),
        ("httpx.get", "function"),
        ("asyncio.run", "function"),
        ("json", "method"),
        ("raise_for_status", "method"),
        ("AsyncClient", "class"),
        ("BasicAuth", "class"),
        ("follow_redirects", "error_code"),
        ("card_declined", "error_code"),
        ("rate_limit_exceeded", "error_code"),
        ("GEMINI_API_KEY", "env_var"),
        ("HTTP_PROXY", "env_var"),
        ("SECRET", "env_var"),
    ]
    for pair in expected_pairs:
        assert pair in pairs, f"Missing extracted identifier: {pair} in {pairs}"

    # No span double-extraction: DELETE from endpoint not re-extracted as env_var;
    # raise_for_status() method not re-extracted as error_code.
    assert ("DELETE", "env_var") not in pairs
    assert ("raise_for_status", "error_code") not in pairs
    assert ("Client", "class") not in pairs

    # Context window is populated (~80 chars centered around match)
    for item in extracted:
        assert isinstance(item, Identifier)
        assert item.context
        assert len(item.context) <= 80


def test_extract_empty_when_no_identifiers():
    """Plain English markdown without API identifiers returns an empty list."""
    markdown = """
## Overview

This short guide explains general concepts in plain prose with simple words.
"""
    assert extract_identifiers(markdown) == []


def test_extract_deduplicates_preserving_first_appearance_order():
    """Duplicate identifiers are deduplicated by (text, kind), keeping first appearance order."""
    markdown = """
First call `httpx.Client`, then `POST /v1/items`, then `response.json()`,
then `httpx.Client` again, `response.json()` again, and `POST /v1/items`.
"""
    extracted = extract_identifiers(markdown)
    pairs = [(item.text, item.kind) for item in extracted]

    assert pairs == [
        ("httpx.Client", "class"),
        ("POST /v1/items", "endpoint"),
        ("json", "method"),
    ]


def test_extract_english_snake_case_is_allowlisted():
    """English compound tokens like use_case are excluded from unverified via allowlist."""
    markdown = "For each `use_case` and `edge_case`, verify behavior carefully."
    extracted = extract_identifiers(markdown)
    matching = [item for item in extracted if item.text in ("use_case", "edge_case")]
    # Either excluded at extraction OR allowlisted as known_external
    assert all(is_known_external(item) for item in matching)
