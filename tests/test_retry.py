import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from doc2skill.ingest.base import Document
from doc2skill.llm.base import FrontmatterResult
from doc2skill.llm.gemini import GeminiProvider


def test_gemini_retry_429_twice_then_200(caplog):
    """Mock generate_content returning 429 on attempt 1, 429 on attempt 2, and 200 on attempt 3.

    Assert:
    - 3 calls total
    - Final result is valid FrontmatterResult
    """
    mock_client = MagicMock()
    err_429_1 = genai_errors.APIError(429, "Rate limit exceeded (1)", None)
    err_429_2 = genai_errors.APIError(429, "Rate limit exceeded (2)", None)

    mock_resp_200 = MagicMock()
    mock_resp_200.text = json.dumps(
        {
            "description": "Successfully synthesized after retries.",
            "when_to_use": "- When retry succeeds on third attempt",
        }
    )
    mock_resp_200.usage_metadata.prompt_token_count = 120
    mock_resp_200.usage_metadata.candidates_token_count = 30

    mock_client.models.generate_content.side_effect = [
        err_429_1,
        err_429_2,
        mock_resp_200,
    ]

    provider = GeminiProvider(api_key="test-key", client=mock_client)

    docs = [
        Document(
            path="guide.md",
            content="# Guide\nDetailed guide documentation.",
            title="Guide",
        )
    ]

    with (
        patch("time.sleep"),
        caplog.at_level(logging.WARNING),
    ):
        result = provider.synthesize_frontmatter(docs)

    # Assert 3 calls total
    assert mock_client.models.generate_content.call_count == 3
    # Assert final result is valid
    assert isinstance(result, FrontmatterResult)
    assert result.description == "Successfully synthesized after retries."
    assert "- When retry succeeds on third attempt" in result.when_to_use
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    # Assert warning logs for attempt 1 and 2
    assert "Gemini retry 1/4" in caplog.text
    assert "Gemini retry 2/4" in caplog.text


def test_gemini_no_retry_on_400():
    """Mock generate_content returning 400 once.

    Assert NO retry happens (fails fast with call_count == 1).
    """
    mock_client = MagicMock()
    err_400 = genai_errors.APIError(400, "Bad Request: Invalid argument", None)
    mock_client.models.generate_content.side_effect = err_400

    provider = GeminiProvider(api_key="test-key", client=mock_client)

    docs = [
        Document(
            path="guide.md",
            content="# Guide\nDetailed guide documentation.",
            title="Guide",
        )
    ]

    with pytest.raises(genai_errors.APIError) as exc_info:
        provider.synthesize_frontmatter(docs)

    assert exc_info.value.code == 400
    # Assert fails fast on first attempt without retrying
    assert mock_client.models.generate_content.call_count == 1
