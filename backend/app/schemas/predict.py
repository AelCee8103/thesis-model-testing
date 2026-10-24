"""Pydantic schemas for POST /predict."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10_000,
        description="Input text. Max 10,000 chars.",
    )
    dataset: str = Field(..., description="Dataset the model was trained on.")
    dimension: str = Field(..., description="'valence' or 'arousal'.")
    architecture: str = Field("bilstm", description="'lstm' or 'bilstm'.")
    run_tag: str = Field("paper", description="Training config: 'paper'.")


class PredictResponse(BaseModel):
    score: float
    score_raw: float
    scale_min: float
    scale_max: float
    dataset: str
    dimension: str
    architecture: str
    run_tag: str
    region_depth: int
    region_capacity: int
    n_regions: int
    n_tokens_total: int
    regions: List[List[str]]
    warnings: List[str]
