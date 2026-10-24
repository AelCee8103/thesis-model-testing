"""Pydantic schemas for POST /evaluate — ground-truth metrics for both models."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    truth: List[float] = Field(
        ..., min_length=1, description="Ground-truth scores, one per row.",
    )
    baseline_pred: List[float] = Field(
        ..., min_length=1, description="Baseline (LSTM) predictions, aligned with truth.",
    )
    proposed_pred: List[float] = Field(
        ..., min_length=1, description="Proposed (BiLSTM) predictions, aligned with truth.",
    )
    dimension: str = Field(..., description="'valence' or 'arousal' (echoed back).")


class MetricSet(BaseModel):
    mae: float
    pearson_r: float


class MannWhitneyResult(BaseModel):
    u_statistic: float
    p_value: float
    significant: bool


class EvaluateResponse(BaseModel):
    n: int
    dimension: str
    baseline: MetricSet
    proposed: MetricSet
    mann_whitney_u: MannWhitneyResult
