"""FastAPI dependencies — shared singletons."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.cache import ResourceCache


@lru_cache
def get_cache() -> ResourceCache:
    """Single ResourceCache shared across all requests in this process."""
    return ResourceCache(get_settings().base_dir)
