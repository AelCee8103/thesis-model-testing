"""GET /models — enumerate available checkpoints on disk."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.model_info import ModelInfo
from app.services.discovery import discover_models

router = APIRouter(tags=["models"])


@router.get("/models", response_model=List[ModelInfo])
def list_models(settings: Settings = Depends(get_settings)) -> List[dict]:
    return discover_models(settings.base_dir)
