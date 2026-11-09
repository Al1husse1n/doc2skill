import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "doc2skill"


def get_cache_dir(custom_dir: Path | str | None = None) -> Path:
    """Return cache directory, creating it if it does not exist."""
    if custom_dir:
        target = Path(custom_dir)
    elif os.environ.get("DOC2SKILL_CACHE_DIR"):
        target = Path(os.environ["DOC2SKILL_CACHE_DIR"])
    else:
        target = DEFAULT_CACHE_DIR

    target.mkdir(parents=True, exist_ok=True)
    return target


def cache_key(
    source: str,
    max_tokens: int = 50_000,
    model: str = "gemini-2.5-flash",
    source_type: str = "markdown_dir",
) -> str:
    """Compute SHA256 of (canonical_source, max_tokens, model, source_type)."""
    if source_type == "url" or source.startswith(("http://", "https://")):
        canonical = source.strip()
    else:
        canonical = str(Path(source).resolve())
    data = f"{canonical}:{max_tokens}:{model}:{source_type}".encode()
    return hashlib.sha256(data).hexdigest()


def get(key: str, cache_dir: Path | str | None = None) -> str | None:
    """Check ~/.cache/doc2skill/<key>.json and return content string or None."""
    folder = get_cache_dir(cache_dir)
    cache_file = folder / f"{key}.json"
    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data.get("content")
    except (json.JSONDecodeError, OSError):
        return None


def set(key: str, content: str, cache_dir: Path | str | None = None) -> None:
    """Write cache JSON file with timestamp."""
    folder = get_cache_dir(cache_dir)
    cache_file = folder / f"{key}.json"
    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "content": content,
    }
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
