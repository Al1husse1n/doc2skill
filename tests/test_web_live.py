import json
import os

import httpx
import pytest

from doc2skill.web.app import create_app


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live ASGI web tests against network and Gemini",
)
@pytest.mark.asyncio
async def test_live_build_example_com():
    """ASGI client, real network fetch of https://example.com/, real Gemini synthesis."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as client:
        resp = await client.post(
            "/api/build",
            json={
                "source": "https://example.com/",
                "source_type": "url",
                "max_pages": 1,
                "max_tokens": 50_000,
                "no_cache": True,
                "no_llm": False,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    blocks = [b for b in resp.text.split("\n\n") if b.strip()]
    parsed = []
    for block in blocks:
        ev_name = ""
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            parsed.append((ev_name, json.loads("\n".join(data_lines))))

    names = [name for name, _ in parsed]
    assert "complete" in names, f"Expected complete event, got events: {parsed}"
    complete_data = dict(parsed)["complete"]
    assert "example" in complete_data["skill_md"].lower()
