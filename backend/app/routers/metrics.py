"""POST /evaluate — MAE, Pearson r, and a Mann-Whitney U test on absolute errors."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.metrics import EvaluateRequest, EvaluateResponse
from app.services.metrics import compute_evaluation_metrics

log = logging.getLogger("tsr-backend.routes.metrics")

router = APIRouter(tags=["metrics"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> dict:
    try:
        return compute_evaluation_metrics(
            truth=req.truth,
            baseline_pred=req.baseline_pred,
            proposed_pred=req.proposed_pred,
            dimension=req.dimension,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        log.exception("evaluate failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
