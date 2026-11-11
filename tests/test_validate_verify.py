from doc2skill.validate.extract import Identifier, extract_identifiers
from doc2skill.validate.verify import ValidationReport, verify_identifiers


def test_verify_known_class_in_source():
    """Source contains httpx.Client -> classified as verified."""
    skill_md = "Create a synchronous client via `httpx.Client()`."
    source = "The `httpx.Client` class manages connection pooling."

    report = verify_identifiers(extract_identifiers(skill_md), source)
    assert isinstance(report, ValidationReport)
    assert [i.text for i in report.verified] == ["httpx.Client"]
    assert report.unverified == []
    assert report.precision == 1.0


def test_verify_unverified_custom_class():
    """Source has httpx.Client while skill hallucinates httpx.CustomClient -> unverified."""
    skill_md = "Use `httpx.Client` or `httpx.CustomClient`."
    source = "Use `httpx.Client` for requests."

    report = verify_identifiers(extract_identifiers(skill_md), source)
    assert [i.text for i in report.verified] == ["httpx.Client"]
    assert [i.text for i in report.unverified] == ["httpx.CustomClient"]
    assert report.precision == 0.5


def test_verify_error_code_and_method_in_source():
    """follow_redirects and .json() method verify against source corpus."""
    skill_md = "Set `follow_redirects=True` and call `response.json()`."
    source = "Pass `follow_redirects=True` on redirects. Parse JSON with `r.json()`."

    report = verify_identifiers(extract_identifiers(skill_md), source)
    verified_texts = {i.text for i in report.verified}
    assert verified_texts == {"follow_redirects", "json"}
    assert report.unverified == []
    assert report.precision == 1.0


def test_verify_asyncio_run_classified_as_known_external():
    """asyncio.run absent from source goes to known_external, excluded from precision denominator."""
    skill_md = "Run async workflow with `asyncio.run(main())` and `httpx.AsyncClient`."
    source = "Use `httpx.AsyncClient` for async HTTP requests."

    report = verify_identifiers(extract_identifiers(skill_md), source)
    assert [i.text for i in report.known_external] == ["asyncio.run"]
    assert [i.text for i in report.verified] == ["httpx.AsyncClient"]
    assert report.unverified == []
    assert report.precision == 1.0


def test_verify_empty_source_marks_non_allowlisted_unverified():
    """Empty source marks domain identifiers unverified while keeping stdlib in known_external."""
    skill_md = (
        "Use `httpx.Client`, `POST /v1/charges`, `asyncio.run(main())`, "
        "`use_case`, and `GEMINI_API_KEY`."
    )
    report = verify_identifiers(extract_identifiers(skill_md), "")

    assert report.verified == []
    assert {i.text for i in report.known_external} == {"asyncio.run", "use_case"}
    assert {i.text for i in report.unverified} == {
        "httpx.Client",
        "POST /v1/charges",
        "GEMINI_API_KEY",
    }
    assert report.precision == 0.0


def test_bare_class_verification():
    """Dotted classes documented under bare headings or class defs verify with context='bare'."""
    ids = [Identifier("httpx.URL", "class", "")]
    corpus = "## URL\n\nclass URL(url: str, **kwargs): ..."
    report = verify_identifiers(ids, corpus)
    assert len(report.verified) == 1
    assert report.verified[0].context == "bare"


def test_bare_class_backtick_heading_and_full_vs_unverified():
    """Backtick markdown headings like ## `Headers` verify bare; prose-only bare words stay unverified."""
    ids = [
        Identifier("httpx.Headers", "class", ""),
        Identifier("httpx.Client", "class", ""),
        Identifier("httpx.FakeClient", "class", ""),
    ]
    corpus = (
        "## `Headers`\n\nHeader container.\n\n"
        "Use `httpx.Client` for requests; FakeClient appears only as a plain prose word."
    )
    report = verify_identifiers(ids, corpus)
    verified_contexts = {item.text: item.context for item in report.verified}
    assert verified_contexts == {"httpx.Headers": "bare", "httpx.Client": "full"}
    assert [item.text for item in report.unverified] == ["httpx.FakeClient"]


def test_source_divergent_defaults_and_signature_bounds():
    """DEFAULT_* constants and (max|min)_*/pool_limits/max_redirects in signatures go to source_divergent."""
    skill_md = """
```python
httpx.Client(
    timeout=DEFAULT_TIMEOUT,
    pool_limits: Limits = DEFAULT_LIMITS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
)
```
In prose, `min_retry_count` without signature syntax is unverified.
"""
    source = "Use `httpx.Client` for requests."
    report = verify_identifiers(extract_identifiers(skill_md), source)

    assert [i.text for i in report.verified] == ["httpx.Client"]
    assert {i.text for i in report.source_divergent} == {
        "DEFAULT_TIMEOUT",
        "pool_limits",
        "DEFAULT_LIMITS",
        "max_redirects",
        "DEFAULT_MAX_REDIRECTS",
    }
    assert [i.text for i in report.unverified] == ["min_retry_count"]
    # precision excludes source_divergent (1 / (1 + 1) = 0.5)
    assert report.precision == 0.5
    # inference_rate = 5 / (1 + 5 + 1) = 5/7
    assert abs(report.inference_rate - (5 / 7)) < 1e-9

