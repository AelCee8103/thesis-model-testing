"""Application settings loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Path that contains models/, cache/, and the dataset CSVs.
    base_dir: Path = _BACKEND_ROOT

    host: str = "127.0.0.1"
    port: int = 8000

    cors_origins: List[str] = ["*"]

    model_config = {
        "env_file": str(_BACKEND_ROOT / ".env"),
        "env_prefix": "",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
