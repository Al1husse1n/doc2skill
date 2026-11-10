from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from doc2skill.distill.condense import CondensedResult, build_condense_prompt, condense
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

        with pytest.raises(ValueError, match="suspiciously short"):
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
