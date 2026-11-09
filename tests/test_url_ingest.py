from unittest.mock import AsyncMock, patch

import httpx
import pytest

from doc2skill.ingest.base import Document
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
    assert "# Example Domain" in doc.content

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


@pytest.mark.asyncio
async def test_sitemap_url_ingestion():
    """Verify that a sitemap URL discovers and fetches pages."""
    mock_urls = [
        "https://test.com/guide1",
        "https://test.com/guide2",
    ]
    docs_map = {
        "https://test.com/guide1": Document(
            path="https://test.com/guide1",
            content="# Guide 1\nContent 1",
            title="Guide 1",
        ),
        "https://test.com/guide2": Document(
            path="https://test.com/guide2",
            content="# Guide 2\nContent 2",
            title="Guide 2",
        ),
    }

    async def fake_extract(url, client):
        return docs_map.get(url)

    with (
        patch("doc2skill.ingest.url.discover_sitemap_urls", return_value=mock_urls),
        patch("doc2skill.ingest.url.extract_page", side_effect=fake_extract),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        docs = await load_url("https://test.com/sitemap.xml", max_pages=10)
        assert len(docs) == 2
        assert docs[0].path == "https://test.com/guide1"
        assert docs[1].path == "https://test.com/guide2"


@pytest.mark.asyncio
async def test_sitemap_max_pages_truncation(caplog):
    """Verify sitemap discovery is capped by max_pages and logs a warning."""
    import logging

    mock_urls = [f"https://test.com/page{i}" for i in range(10)]

    async def fake_extract(url, client):
        return Document(path=url, content="content", title="title")

    with (
        patch("doc2skill.ingest.url.discover_sitemap_urls", return_value=mock_urls),
        patch("doc2skill.ingest.url.extract_page", side_effect=fake_extract),
        patch("asyncio.sleep", new_callable=AsyncMock),
        caplog.at_level(logging.WARNING),
    ):
        docs = await load_url("https://test.com/sitemap.xml", max_pages=3)
        assert len(docs) == 3
        assert "Truncated to 3 pages (found 10)" in caplog.text


@pytest.mark.asyncio
async def test_sitemap_robots_disallowed_skipped():
    """Verify that disallowed pages in a sitemap are skipped without failing the run."""
    mock_urls = [
        "https://test.com/allowed",
        "https://test.com/forbidden",
    ]

    async def fake_extract(url, client):
        if "forbidden" in url:
            return None
        return Document(path=url, content="allowed content", title="Allowed")

    with (
        patch("doc2skill.ingest.url.discover_sitemap_urls", return_value=mock_urls),
        patch("doc2skill.ingest.url.extract_page", side_effect=fake_extract),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        docs = await load_url("https://test.com/sitemap.xml", max_pages=10)
        assert len(docs) == 1
        assert docs[0].path == "https://test.com/allowed"


@pytest.mark.asyncio
async def test_trafilatura_heading_preservation():
    """Verify that a page with <h1> + <h2> + <p> produces markdown with # and ## headings."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    html_content = """<!doctype html>
<html>
<head><title>Documentation Guide</title></head>
<body>
<nav><ul><li><a href="/">Nav Item</a></li></ul></nav>
<main>
<h1>Architecture Overview</h1>
<p>Here is introductory text explaining the system architecture.</p>
<h2>Subsystem Details</h2>
<p>Here are specific details about the subsystem components.</p>
</main>
<footer><p>Copyright 2026 Company</p></footer>
</body>
</html>"""
    page_resp = httpx.Response(
        200,
        text=html_content,
        request=httpx.Request("GET", "https://test.com/guide"),
    )
    mock_client.get.return_value = page_resp

    with patch("doc2skill.ingest.url.can_fetch", return_value=True):
        doc = await extract_page("https://test.com/guide", mock_client)

    assert doc is not None
    # Heading preservation in markdown
    assert "# Architecture Overview" in doc.content
    assert "## Subsystem Details" in doc.content
    # Boilerplate stripping
    assert "Nav Item" not in doc.content
    assert "Copyright 2026" not in doc.content


@pytest.mark.asyncio
async def test_sitemap_fallback_to_single_page():
    """Verify that a 404 on /sitemap.xml falls through to single-page ingestion mode."""
    requested_urls: list[tuple[str, str]] = []

    async def mock_transport(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        requested_urls.append((request.method, url_str))

        if request.method == "HEAD" and url_str == "https://test.com/sitemap.xml":
            return httpx.Response(404, request=request)
        if url_str == "https://test.com/robots.txt":
            return httpx.Response(404, request=request)
        if url_str == "https://test.com/docs/single-page":
            html = """<html>
            <head><title>Single Page Title</title></head>
            <body>
            <h1>Single Page Title</h1>
            <p>Single page body content to extract.</p>
            </body></html>"""
            return httpx.Response(200, text=html, request=request)

        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(mock_transport)
    client = httpx.AsyncClient(transport=transport)

    with patch("doc2skill.ingest.url._robots_cache", {}):
        docs = await load_url(
            "https://test.com/docs/single-page",
            follow_sitemap=True,
            client=client,
        )

    # 1. Verify HEAD to /sitemap.xml occurred and returned 404
    head_sitemap_calls = [
        (method, url)
        for method, url in requested_urls
        if method == "HEAD" and "/sitemap.xml" in url
    ]
    assert len(head_sitemap_calls) == 1

    # 2. Verify single page was requested and returned 1 Document
    assert len(docs) == 1
    assert docs[0].path == "https://test.com/docs/single-page"
    assert "Single Page Title" in docs[0].title
    assert "Single page body content to extract" in docs[0].content
