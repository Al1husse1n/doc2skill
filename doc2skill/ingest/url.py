import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
import usp.tree

from doc2skill.ingest.base import Document

logger = logging.getLogger(__name__)

# User-agent header for polite requests
# TODO: update with real repository URL
USER_AGENT = "doc2skill/0.1 (+https://github.com/doc2skill/doc2skill)"
POLITE_DELAY = 0.2
DEFAULT_TIMEOUT = 10.0

_robots_cache: dict[str, RobotFileParser] = {}


def extract_html_title(html: str) -> str | None:
    """Extract <title> content from raw HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        if title:
            # Clean inner whitespace/newlines
            return " ".join(title.split())
    return None


def derive_url_title(html: str, markdown: str, url: str) -> str:
    """Derive page title: <title> tag -> first markdown # -> URL path."""
    html_title = extract_html_title(html)
    if html_title:
        return html_title

    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            heading = match.group(1).strip()
            if heading:
                return heading

    parsed = urlparse(url)
    clean_path = parsed.path.strip("/")
    if clean_path:
        return Path(clean_path).name or clean_path
    return parsed.netloc or url


async def can_fetch(url: str, client: httpx.AsyncClient) -> bool:
    """Check if the URL is permitted to be fetched according to robots.txt."""
    parsed = urlparse(url)
    domain_root = f"{parsed.scheme}://{parsed.netloc}"

    if domain_root in _robots_cache:
        rp = _robots_cache[domain_root]
        return rp.can_fetch(USER_AGENT, url)

    rp = RobotFileParser()
    robots_url = urljoin(domain_root, "/robots.txt")
    try:
        resp = await client.get(robots_url, timeout=3.0)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            # Blocked access to robots.txt -> disallow by default
            _robots_cache[domain_root] = rp
            return False
        else:
            # 404 or other -> allow by default
            rp.allow_all = True
    except (httpx.HTTPError, OSError):
        rp.allow_all = True

    _robots_cache[domain_root] = rp
    return rp.can_fetch(USER_AGENT, url)


async def extract_page(url: str, client: httpx.AsyncClient) -> Document | None:
    """Fetch URL, perform robots.txt check, extract main markdown content."""
    allowed = await can_fetch(url, client)
    if not allowed:
        logger.warning("Skipped (robots.txt): %s", url)
        return None

    try:
        resp = await client.get(url, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
        if resp.status_code >= 400:
            logger.warning("Skipped (HTTP %d): %s", resp.status_code, url)
            return None
        html = resp.text
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Skipped (network error %s): %s", exc, url)
        return None

    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_formatting=True,
        include_links=False,
        include_images=False,
    )

    if not extracted or not extracted.strip():
        logger.warning("Skipped (no extractable content): %s", url)
        return None

    title = derive_url_title(html, extracted, url)

    # Prepend derived title if extracted markdown does not already contain a top-level heading
    clean_extracted = extracted.strip()
    if not re.search(r"^#+\s+", clean_extracted, re.MULTILINE):
        clean_extracted = f"# {title}\n\n{clean_extracted}"

    return Document(
        path=url,
        content=clean_extracted,
        title=title,
        metadata={"source_type": "url", "url": url},
    )


async def check_sitemap_exists(domain: str, client: httpx.AsyncClient) -> bool:
    """Check existence of /sitemap.xml with a fast HEAD request."""
    sitemap_url = urljoin(domain, "/sitemap.xml")
    try:
        resp = await client.head(sitemap_url, timeout=3.0, follow_redirects=True)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def discover_sitemap_urls(
    domain_or_url: str, extra_paths: set[str] | None = None
) -> list[str]:
    """Discover URLs from sitemaps using ultimate-sitemap-parser."""
    try:
        tree = usp.tree.sitemap_tree_for_homepage(
            domain_or_url, extra_known_paths=extra_paths
        )
        urls: list[str] = []
        for page in tree.all_pages():
            if getattr(page, "url", None):
                urls.append(page.url)
        return urls
    except Exception as exc:
        logger.warning("Failed to discover sitemap URLs for %s: %s", domain_or_url, exc)
        return []


async def load_url(
    url: str,
    max_pages: int = 100,
    follow_sitemap: bool = True,
    client: httpx.AsyncClient | None = None,
) -> list[Document]:
    """Ingest content from a single URL or expand via sitemap.

    Returns a list of Documents sorted by URL.
    """
    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        should_close_client = True

    try:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        is_sitemap = url.endswith(".xml") or "sitemap" in parsed.path.lower()

        if not is_sitemap and follow_sitemap:
            has_sitemap = await check_sitemap_exists(domain, client)
            if has_sitemap:
                is_sitemap = True

        documents: list[Document] = []

        if is_sitemap:
            # Discover URLs via sitemap tree in thread pool
            loop = asyncio.get_running_loop()
            extra_paths = (
                {parsed.path} if parsed.path and parsed.path.endswith(".xml") else None
            )
            discovered = await loop.run_in_executor(
                None, discover_sitemap_urls, domain, extra_paths
            )
            total = len(discovered)
            if total > max_pages:
                logger.warning("Truncated to %d pages (found %d)", max_pages, total)
                discovered = discovered[:max_pages]

            for page_url in discovered:
                doc = await extract_page(page_url, client)
                if doc:
                    documents.append(doc)
                await asyncio.sleep(POLITE_DELAY)

            if not documents and not url.endswith(".xml"):
                doc = await extract_page(url, client)
                if doc:
                    documents.append(doc)
        else:
            doc = await extract_page(url, client)
            if doc:
                documents.append(doc)

        documents.sort(key=lambda d: d.path)
        return documents
    finally:
        if should_close_client:
            await client.aclose()
