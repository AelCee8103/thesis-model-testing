"""POST /predict — run a single-text prediction."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_cache
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.cache import ResourceCache
from app.services.predictor import predict_text

log = logging.getLogger("tsr-backend.routes.predict")

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    cache: ResourceCache = Depends(get_cache),
) -> dict:
    try:
        return predict_text(
            text=req.text,
            dataset=req.dataset,
            dimension=req.dimension,
            architecture=req.architecture,
            run_tag=req.run_tag,
            cache=cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
