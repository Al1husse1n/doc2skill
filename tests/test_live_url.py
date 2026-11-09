import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from doc2skill.cli import app
from doc2skill.ingest.url import load_url

runner = CliRunner()


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live URL tests against the network",
)
@pytest.mark.asyncio
async def test_live_url_ingest():
    """Verify live HTTP fetch and content extraction against example.com."""
    docs = await load_url("https://example.com/", follow_sitemap=False)
    assert len(docs) == 1
    assert docs[0].path == "https://example.com/"
    assert "documentation examples" in docs[0].content
    assert docs[0].title == "Example Domain"


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live URL tests against the network",
)
def test_live_url_cli_build(tmp_path: Path):
    """Verify live CLI build against example.com."""
    output_dir = tmp_path / "live_out"
    result = runner.invoke(
        app,
        [
            "https://example.com/",
            "--output",
            str(output_dir),
            "--no-llm",
            "--no-cache",
            "--no-sitemap",
        ],
    )
    assert result.exit_code == 0
    assert "Source: https://example.com/ (single page)" in result.output
    assert "Documents loaded: 1" in result.output

    emitted_skill = output_dir / "example" / "SKILL.md"
    assert emitted_skill.exists()
    content = emitted_skill.read_text(encoding="utf-8")
    assert "name: example" in content
    assert "Example Domain" in content or "documentation examples" in content
