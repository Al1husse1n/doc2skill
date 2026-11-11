import logging
import os
from typing import Any

import httpx

from doc2skill.cache.disk import cache_key

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "skill_cache"
DEFAULT_TIMEOUT = 5.0


def _get_config() -> tuple[str, str, str]:
    """Read Supabase REST credentials and table name from environment variables."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    table = os.environ.get("SUPABASE_CACHE_TABLE", DEFAULT_TABLE).strip() or DEFAULT_TABLE
    return url, key, table


def is_supabase_configured() -> bool:
    """Return True when both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set."""
    url, key, _ = _get_config()
    return bool(url and key)


def get_supabase_client() -> httpx.Client | None:
    """Create a configured Supabase PostgREST httpx.Client, or None if unconfigured."""
    url, key, _ = _get_config()
    if not url or not key:
        return None
    base_url = f"{url.rstrip('/')}/rest/v1"
    return httpx.Client(
        base_url=base_url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        timeout=DEFAULT_TIMEOUT,
    )


def _infer_source_type(source: str) -> str:
    return "url" if source.startswith(("http://", "https://")) else "markdown_dir"


def get_cached_skill(
    source: str,
    max_tokens: int,
    model: str,
    *,
    client: Any | None = None,
) -> str | None:
    """Look up cached SKILL.md from Supabase skill_cache table, or return None on miss/unconfigured."""
    owned_client = False
    if client is None:
        client = get_supabase_client()
        if client is None:
            return None
        owned_client = True

    _, _, table = _get_config()
    key = cache_key(
        source,
        max_tokens=max_tokens,
        model=model,
        source_type=_infer_source_type(source),
    )

    try:
        resp = client.get(
            f"/{table}",
            params={
                "cache_key": f"eq.{key}",
                "select": "skill_md",
                "limit": "1",
            },
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list) and rows:
            val = rows[0].get("skill_md")
            return str(val) if isinstance(val, str) else None
        return None
    except (httpx.HTTPError, ValueError, OSError, KeyError, TypeError) as exc:
        logger.warning("Supabase cache get failed (%s); falling back to disk cache.", exc)
        return None
    finally:
        if owned_client and hasattr(client, "close"):
            client.close()


def set_cached_skill(
    source: str,
    max_tokens: int,
    model: str,
    skill_md: str,
    *,
    client: Any | None = None,
) -> None:
    """Upsert generated SKILL.md into Supabase skill_cache table (no-op if unconfigured)."""
    if not skill_md:
        return

    owned_client = False
    if client is None:
        client = get_supabase_client()
        if client is None:
            return
        owned_client = True

    _, _, table = _get_config()
    key = cache_key(
        source,
        max_tokens=max_tokens,
        model=model,
        source_type=_infer_source_type(source),
    )

    try:
        resp = client.post(
            f"/{table}",
            params={"on_conflict": "cache_key"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            json={
                "cache_key": key,
                "source": source,
                "max_tokens": max_tokens,
                "model": model,
                "skill_md": skill_md,
            },
        )
        resp.raise_for_status()
    except (httpx.HTTPError, ValueError, OSError, TypeError) as exc:
        logger.warning("Supabase cache set failed (%s); local cache unaffected.", exc)
    finally:
        if owned_client and hasattr(client, "close"):
            client.close()

