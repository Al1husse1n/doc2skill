import json
from pathlib import Path

import pytest

from doc2skill.cache import disk as cache


def test_cache_set_and_get(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Assert set then get returns same content, and get on missing key returns None."""
    # Use isolated cache directory
    monkeypatch.setenv("DOC2SKILL_CACHE_DIR", str(tmp_path))

    key = "sample_test_key"
    content = "# Sample Skill Content\n\nCached markdown text."

    # 1. Missing key returns None
    assert cache.get("missing_key") is None

    # 2. Set key then get returns same content
    cache.set(key, content)
    retrieved = cache.get(key)
    assert retrieved == content

    # 3. Verify JSON file structure on disk
    cache_file = tmp_path / f"{key}.json"
    assert cache_file.exists()
    raw_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "timestamp" in raw_data
    assert raw_data["content"] == content


def test_cache_key_generation():
    """Verify cache_key produces SHA256 hex string based on abs_path and config_hash."""
    path1 = "docs/guide"
    key1 = cache.cache_key(path1, "config_a")
    key2 = cache.cache_key(path1, "config_a")
    key3 = cache.cache_key(path1, "config_b")

    # Deterministic for same inputs
    assert key1 == key2
    # 64 hex characters (SHA256)
    assert len(key1) == 64
    # Different configs yield different keys
    assert key1 != key3
