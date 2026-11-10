import httpx
import pytest

from doc2skill.web.app import create_app


@pytest.mark.asyncio
async def test_health():
    """GET /api/health returns 200 with status ok."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_serves_html():
    """GET / returns 200 HTML containing the form and doc2skill branding."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<form" in resp.text
    assert "doc2skill" in resp.text
