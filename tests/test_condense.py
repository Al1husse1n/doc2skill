import logging
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from doc2skill.distill.condense import CondensedResult, condense
from doc2skill.ingest.base import Document


def _make_sample_docs() -> list[Document]:
    return [
        Document(
            path="https://docs.example.com/client",
            content="# Client\n" + ("Use Client.get() with explicit timeout. " * 40),
            title="Client",
        ),
        Document(
            path="https://docs.example.com/async",
            content="# Async\n" + ("Use AsyncClient.get() inside async functions. " * 40),
            title="Async",
        ),
    ]


@pytest.mark.asyncio
async def test_condense_success_and_fence_stripping():
    """Mock LLM returning fenced markdown body and verify CondensedResult fields."""
    mock_client = MagicMock()
    fixture_body = (
        "## API surface\n`Client.get(url)`\n\n"
        "## Common workflows\n1. Instantiate `Client()`.\n\n"
        "## Gotchas\nAlways configure timeout.\n\n"
        "## Reference\nSee client module."
    )
    mock_resp = MagicMock()
    mock_resp.text = f"```markdown\n{fixture_body}\n```"
    mock_resp.usage_metadata.prompt_token_count = 1200
    mock_resp.usage_metadata.candidates_token_count = 650
    mock_client.models.generate_content.return_value = mock_resp

    docs = _make_sample_docs()
    result = await condense(
        documents=docs,
        model="gemini-2.5-flash",
        client=mock_client,
    )

    assert isinstance(result, CondensedResult)
    assert result.body == fixture_body
    assert not result.body.startswith("```")
    assert result.sources == [
        "https://docs.example.com/client",
        "https://docs.example.com/async",
    ]
    assert result.input_tokens == 1200
    assert result.output_tokens == 650
    assert result.dropped_documents == []
    assert result.truncated is False

    # Verify prompt formatting includes delimiters and doc count
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "Total documents in corpus: 2" in call_kwargs["contents"]
    assert "=== DOCUMENT: https://docs.example.com/client ===" in call_kwargs["contents"]
    assert "=== END DOCUMENT ===" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_condense_rejects_empty_or_suspiciously_short_output():
    """Empty or suspiciously short output raises ValueError."""
    docs = _make_sample_docs()

    for bad_text, tok_count in [("", 0), (".", 1), ("I cannot help with that.", 6), ("Short body", 250)]:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = bad_text
        mock_resp.usage_metadata.prompt_token_count = 800
        mock_resp.usage_metadata.candidates_token_count = tok_count
        mock_client.models.generate_content.return_value = mock_resp

        with pytest.raises(
            ValueError,
            match=r"Condensation (produced empty/stub output|refused)",
        ):
            await condense(documents=docs, model="gemini-2.5-flash", client=mock_client)


@pytest.mark.asyncio
async def test_condense_retry_429_then_200():
    """Verify condense retries 429 twice and succeeds on 3rd attempt."""
    mock_client = MagicMock()
    err1 = genai_errors.APIError(429, "Rate limit 1", None)
    err2 = genai_errors.APIError(429, "Rate limit 2", None)
    ok_resp = MagicMock()
    ok_resp.text = "## API surface\n`Client.get()`\n\n## Common workflows\n1. Call get.\n\n## Gotchas\nTimeouts.\n\n## Reference\nclient"
    ok_resp.usage_metadata.prompt_token_count = 900
    ok_resp.usage_metadata.candidates_token_count = 550

    mock_client.models.generate_content.side_effect = [err1, err2, ok_resp]

    with patch("time.sleep"):
        result = await condense(
            documents=_make_sample_docs(),
            model="gemini-2.5-flash",
            client=mock_client,
        )

    assert mock_client.models.generate_content.call_count == 3
    assert result.output_tokens == 550


@pytest.mark.asyncio
async def test_condense_no_retry_on_400():
    """Verify condense fails fast on HTTP 400 without retrying."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.APIError(
        400, "Bad Request", None
    )

    with pytest.raises(genai_errors.APIError):
        await condense(
            documents=_make_sample_docs(),
            model="gemini-2.5-flash",
            client=mock_client,
        )

    assert mock_client.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_condense_truncation_100_docs(caplog):
    """100 fake docs x ~10k tokens exceeds limit -> drops deeply nested tail docs, logs warning."""
    docs: list[Document] = []
    # Mix shallow root docs and deeply nested docs (40_000 chars ~ 10_000 tokens each)
    for i in range(50):
        docs.append(
            Document(
                path=f"https://docs.example.com/core_{i:02d}.md",
                content="A" * 40_000,
                title=f"Core {i}",
            )
        )
        docs.append(
            Document(
                path=f"https://docs.example.com/deep/nested/detail_{i:02d}.md",
                content="B" * 40_000,
                title=f"Detail {i}",
            )
        )

    mock_client = MagicMock()

    def fake_count_tokens(model: str, contents: str):
        resp = MagicMock()
        resp.total_tokens = len(contents) // 4
        return resp

    ok_resp = MagicMock()
    ok_resp.text = (
        "## API surface\n`Core()`\n\n"
        "## Common workflows\n1. Use core methods.\n\n"
        "## Gotchas\nContext limits.\n\n"
        "## Reference\ncore module documentation."
    )
    ok_resp.usage_metadata.prompt_token_count = 120_000
    ok_resp.usage_metadata.candidates_token_count = 12_500

    mock_client.models.count_tokens.side_effect = fake_count_tokens
    mock_client.models.generate_content.return_value = ok_resp

    with caplog.at_level(logging.WARNING):
        result = await condense(
            documents=docs,
            model="gemini-2.5-flash",
            max_context_tokens=250_000,
            client=mock_client,
        )

    assert result.truncated is True
    assert len(result.dropped_documents) > 0
    assert len(result.sources) >= 5
    # Verify all 50 deeply nested documents were dropped BEFORE any shallow core docs
    assert len(result.dropped_documents) > 50
    assert all("/deep/nested/" in path for path in result.dropped_documents[:50])
    assert not any("/deep/nested/" in path for path in result.sources)
    # Verify count_tokens was re-checked after dropping
    assert mock_client.models.count_tokens.call_count >= 2
    assert "Dropped" in caplog.text
    assert "to fit context" in caplog.text


@pytest.mark.asyncio
async def test_condense_truncation_refuses_below_5_docs():
    """If even 5 documents exceed max_context_tokens, raise ValueError without calling LLM."""
    docs = [
        Document(
            path=f"https://docs.example.com/huge_{i}.md",
            content="X" * 400_000,
            title=f"Huge {i}",
        )
        for i in range(5)
    ]

    mock_client = MagicMock()
    count_resp = MagicMock()
    count_resp.total_tokens = 5_000_000
    mock_client.models.count_tokens.return_value = count_resp

    with pytest.raises(ValueError, match="Docs exceed context limit even after truncation"):
        await condense(
            documents=docs,
            model="gemini-2.5-flash",
            max_context_tokens=500_000,
            client=mock_client,
        )

    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_condense_rejects_refusal_markers():
    """LLM refusal markers raise ValueError even if output is non-empty."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "I'm sorry, I can't help with that."
    mock_resp.usage_metadata.prompt_token_count = 500
    mock_resp.usage_metadata.candidates_token_count = 10
    mock_client.models.generate_content.return_value = mock_resp

    with pytest.raises(ValueError, match="Condensation refused"):
        await condense(
            documents=_make_sample_docs(),
            model="gemini-2.5-flash",
            client=mock_client,
        )


@pytest.mark.asyncio
async def test_condense_rejects_stub():
    """Punctuation-only or <100 char stub output raises ValueError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "."
    mock_resp.usage_metadata.prompt_token_count = 500
    mock_resp.usage_metadata.candidates_token_count = 1
    mock_client.models.generate_content.return_value = mock_resp

    with pytest.raises(ValueError, match="Condensation produced empty/stub output"):
        await condense(
            documents=_make_sample_docs(),
            model="gemini-2.5-flash",
            client=mock_client,
        )


@pytest.mark.asyncio
async def test_condense_allows_short_output_for_tiny_input():
    """Tiny 2-doc corpus (<1000 input tokens) succeeds with a valid 150-token skill body."""
    docs = [
        Document(
            path="https://example.com/",
            content="# Example Domain\nThis domain is for use in illustrative examples in documents.",
            title="Example Domain",
        ),
        Document(
            path="https://example.com/about",
            content="# About Example\nReference page describing illustrative example usage.",
            title="About",
        ),
    ]
    short_skill_body = (
        "## API surface\n`GET https://example.com/` — Illustrative HTTP endpoint.\n\n"
        "## Common workflows\n1. Fetch `https://example.com/` for connectivity checks.\n\n"
        "## Gotchas\nNot for production traffic.\n\n"
        "## Reference\nhttps://example.com/"
    )
    assert len(short_skill_body) >= 100

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = short_skill_body
    mock_resp.usage_metadata.prompt_token_count = 500
    mock_resp.usage_metadata.candidates_token_count = 150
    mock_client.models.generate_content.return_value = mock_resp

    result = await condense(
        documents=docs,
        model="gemini-2.5-flash",
        client=mock_client,
    )

    assert result.body == short_skill_body
    assert result.output_tokens == 150


@pytest.mark.asyncio
async def test_condense_rejects_disproportionately_short_output_for_large_input():
    """For input > 1000 tokens, output below 10% of input raises ValueError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = (
        "## API surface\n`Client()`\n\n## Common workflows\n1. Run it.\n\n"
        "## Gotchas\nWatch limits.\n\n## Reference\nSee client docs."
    )
    mock_resp.usage_metadata.prompt_token_count = 5_000
    mock_resp.usage_metadata.candidates_token_count = 200
    mock_client.models.generate_content.return_value = mock_resp

    with pytest.raises(ValueError, match="Condensation too short"):
        await condense(
            documents=_make_sample_docs(),
            model="gemini-2.5-flash",
            client=mock_client,
        )

