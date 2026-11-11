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


def test_supabase_cache_unconfigured_fallback(monkeypatch):
    """When Supabase env vars are missing, helpers safely no-op and return None."""
    from doc2skill.cache import supabase as supabase_cache

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    assert supabase_cache.is_supabase_configured() is False
    assert supabase_cache.get_supabase_client() is None
    assert supabase_cache.get_cached_skill("https://example.com/docs", 50000, "gemini-2.5-flash") is None
    # Should not raise when unconfigured
    supabase_cache.set_cached_skill(
        "https://example.com/docs", 50000, "gemini-2.5-flash", "# SKILL"
    )


def test_supabase_cache_get_and_set_with_mock_client(monkeypatch):
    """Supabase set and get roundtrip against mocked PostgREST transport via get_supabase_client."""
    import json

    import httpx

    from doc2skill.cache import supabase as supabase_cache

    monkeypatch.setenv("SUPABASE_URL", "https://demo.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret-role-key")
    assert supabase_cache.is_supabase_configured() is True

    store: dict[str, dict] = {}
    requests_seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append((request.method, str(request.url)))
        assert request.headers["apikey"] == "secret-role-key"
        assert request.headers["authorization"] == "Bearer secret-role-key"

        if request.method == "POST" and request.url.path == "/rest/v1/skill_cache":
            assert request.url.params.get("on_conflict") == "cache_key"
            assert "resolution=merge-duplicates" in request.headers.get("prefer", "")
            payload = json.loads(request.content.decode("utf-8"))
            store[payload["cache_key"]] = payload
            return httpx.Response(201, json=[])

        if request.method == "GET" and request.url.path == "/rest/v1/skill_cache":
            eq_val = request.url.params.get("cache_key", "")
            assert eq_val.startswith("eq.")
            lookup_key = eq_val[3:]
            row = store.get(lookup_key)
            if row is None:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[{"skill_md": row["skill_md"]}])

        return httpx.Response(404)

    def fake_get_client() -> httpx.Client:
        return httpx.Client(
            base_url="https://demo.supabase.co/rest/v1",
            headers={
                "apikey": "secret-role-key",
                "Authorization": "Bearer secret-role-key",
                "Content-Type": "application/json",
            },
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(supabase_cache, "get_supabase_client", fake_get_client)

    assert (
        supabase_cache.get_cached_skill("https://example.com/docs", 50000, "gemini-2.5-flash")
        is None
    )

    supabase_cache.set_cached_skill(
        "https://example.com/docs",
        50000,
        "gemini-2.5-flash",
        "---\nname: example\n---\n# Cached Skill",
    )

    cached = supabase_cache.get_cached_skill(
        "https://example.com/docs", 50000, "gemini-2.5-flash"
    )
    assert cached == "---\nname: example\n---\n# Cached Skill"
    assert [m for m, _ in requests_seen] == ["GET", "POST", "GET"]


def test_supabase_cache_http_error_returns_none(monkeypatch):
    """Supabase HTTP errors degrade gracefully without throwing so disk fallback continues."""
    import httpx

    from doc2skill.cache import supabase as supabase_cache

    monkeypatch.setenv("SUPABASE_URL", "https://demo.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret-role-key")

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream unavailable"})

    def fake_get_client() -> httpx.Client:
        return httpx.Client(
            base_url="https://demo.supabase.co/rest/v1",
            transport=httpx.MockTransport(failing_handler),
        )

    monkeypatch.setattr(supabase_cache, "get_supabase_client", fake_get_client)

    assert (
        supabase_cache.get_cached_skill("https://example.com/docs", 50000, "gemini-2.5-flash")
        is None
    )
    # set_cached_skill should swallow HTTP errors and log warning instead of crashing
    supabase_cache.set_cached_skill(
        "https://example.com/docs", 50000, "gemini-2.5-flash", "# SKILL"
    )

