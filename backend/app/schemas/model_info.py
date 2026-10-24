"""Pydantic schema for GET /models response entries."""
from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    dataset: str
    dataset_description: str
    language: str
    dimension: str
    architecture: str
    run_tag: str
    scale_min: float
    scale_max: float
    region_depth: int
    checkpoint_exists: bool
