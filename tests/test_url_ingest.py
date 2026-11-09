from unittest.mock import AsyncMock, patch

import httpx
import pytest

from doc2skill.ingest.url import (
    derive_url_title,
    extract_html_title,
    extract_page,
    load_url,
)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_single_page_ingest_example_com():
    """Verify single-page ingestion with trafilatura against example.com (cassette recorded)."""
    docs = await load_url("https://example.com/", follow_sitemap=False)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.path == "https://example.com/"
    assert "documentation examples" in doc.content
    assert doc.title == "Example Domain"

    # Verify HTML boilerplate was stripped by trafilatura
    assert "<!doctype html>" not in doc.content.lower()
    assert "<html" not in doc.content.lower()
    assert "<head" not in doc.content.lower()


@pytest.mark.asyncio
async def test_robots_txt_disallowed_skipped():
    """Verify that a URL disallowed by robots.txt is skipped."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    # robots.txt returns Disallow: /private
    robots_resp = httpx.Response(
        200,
        text="User-agent: *\nDisallow: /private/\n",
        request=httpx.Request("GET", "https://test.com/robots.txt"),
    )
    mock_client.get.return_value = robots_resp

    with patch("doc2skill.ingest.url._robots_cache", {}):
        doc = await extract_page("https://test.com/private/secret.html", mock_client)
        assert doc is None


@pytest.mark.asyncio
async def test_unextractable_content_skipped():
    """Verify pages with no extractable body content are skipped."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    # Empty body page
    page_resp = httpx.Response(
        200,
        text="<html><head><title>Empty</title></head><body></body></html>",
        request=httpx.Request("GET", "https://test.com/empty.html"),
    )
    with patch("doc2skill.ingest.url.can_fetch", return_value=True):
        mock_client.get.return_value = page_resp
        doc = await extract_page("https://test.com/empty.html", mock_client)
        assert doc is None


def test_derive_url_title_fallbacks():
    """Test title extraction priority: <title> -> markdown # -> URL path."""
    html_with_title = "<html><head><title>Custom Web Title</title></head></html>"
    assert extract_html_title(html_with_title) == "Custom Web Title"
    assert (
        derive_url_title(html_with_title, "# Heading", "https://site.com/page")
        == "Custom Web Title"
    )

    # Fallback to first # heading
    html_without_title = "<html><body><h1>Markdown Heading</h1></body></html>"
    assert (
        derive_url_title(
            html_without_title, "# Markdown Heading\nText", "https://site.com/page"
        )
        == "Markdown Heading"
    )

    # Fallback to URL path
    assert (
        derive_url_title("", "no heading here", "https://site.com/docs/guide")
        == "guide"
    )
