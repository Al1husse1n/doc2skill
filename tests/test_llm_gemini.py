import json
from unittest.mock import MagicMock

import pytest

from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult
from doc2skill.llm.gemini import GeminiProvider


def test_gemini_provider_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Assert missing GEMINI_API_KEY raises a clear RuntimeError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(
        RuntimeError, match="GEMINI_API_KEY not set. Export it before running."
    ):
        GeminiProvider()


def test_gemini_provider_synthesize_mock_sdk():
    """Prove GeminiProvider returns a valid FrontmatterResult using a mocked SDK client."""
    mock_client = MagicMock()

    # Setup mock response for generate_content
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "description": "Stripe payments and billing agent skill.",
            "when_to_use": "- When processing card payments\n- When fetching customer records",
        }
    )
    mock_response.usage_metadata.prompt_token_count = 150
    mock_response.usage_metadata.candidates_token_count = 55
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(
        api_key="test-key", model="gemini-2.5-flash", client=mock_client
    )

    docs = [
        Document(
            path="charges.md",
            content="# Charges\nCreate and capture credit card charges.",
            title="Charges API",
        ),
        Document(
            path="customers.md",
            content="# Customers\nManage customer profiles and subscriptions.",
            title="Customers API",
        ),
    ]

    result = provider.synthesize_frontmatter(docs, max_tokens=500)

    assert isinstance(result, FrontmatterResult)
    assert result.description == "Stripe payments and billing agent skill."
    assert "- When processing card payments" in result.when_to_use
    assert result.input_tokens == 150
    assert result.output_tokens == 55

    # Verify generate_content was called with correct model
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


def test_gemini_provider_count_tokens():
    """Verify count_tokens returns SDK count and falls back gracefully on failure."""
    mock_client = MagicMock()
    mock_count_resp = MagicMock()
    mock_count_resp.total_tokens = 42
    mock_client.models.count_tokens.return_value = mock_count_resp

    provider = GeminiProvider(api_key="test-key", client=mock_client)
    assert provider.count_tokens("hello world prompt") == 42

    # Fallback on exception
    mock_client.models.count_tokens.side_effect = RuntimeError("network down")
    fallback = provider.count_tokens("hello world prompt")
    assert fallback == max(1, len("hello world prompt") // 4)


def test_gemini_provider_prompt_truncation():
    """Verify prompt truncates document content to staying within prompt snippet limits."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())
    very_long_doc = "X" * 1500
    docs = [
        Document(
            path="long.md",
            content=very_long_doc,
            title="Long Doc",
        )
    ]
    prompt = provider.build_prompt(docs)
    assert "Long Doc" in prompt
    assert "X" * 1500 not in prompt
    assert "X" * 500 in prompt
