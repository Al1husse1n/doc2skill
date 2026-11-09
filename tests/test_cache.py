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
    """Verify cache_key produces SHA256 hex string based on abs_path, max_tokens, and model."""
    path1 = "docs/guide"
    key1 = cache.cache_key(path1, 50000, "gemini-2.5-flash")
    key2 = cache.cache_key(path1, 50000, "gemini-2.5-flash")
    key3 = cache.cache_key(path1, 50000, "other-model")
    key4 = cache.cache_key(path1, 10000, "gemini-2.5-flash")

    # Deterministic for same inputs
    assert key1 == key2
    # 64 hex characters (SHA256)
    assert len(key1) == 64
    # Different model yields different key
    assert key1 != key3
    # Different max_tokens yields different key
    assert key1 != key4

    # Different source_type yields different key
    key5 = cache.cache_key(path1, 50000, "gemini-2.5-flash", "url")
    assert key1 != key5
