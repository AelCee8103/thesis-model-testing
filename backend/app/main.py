"""FastAPI entrypoint — `uvicorn app.main:app`.

Startup is intentionally fast: we configure GPU, log available checkpoints,
and verify vocab caches exist. We do NOT pre-load Stanza, do NOT pre-warm
models, and do NOT touch any training CSV. All heavy work is lazy on first
/predict request.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

# Import torch BEFORE tensorflow to avoid NCCL symbol conflict when both
# are present in the same process.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, metrics, models, predict
from app.services.discovery import discover_models
from app.utils.gpu import configure_gpu, log_device_info

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tsr-backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_gpu()
    log_device_info()

    available = discover_models(settings.base_dir)
    log.info("[startup] %d checkpoint(s) available at %s/models/", len(available), settings.base_dir)

    # Sanity-check vocab caches without loading them — read-only existence check.
    datasets_with_checkpoints = {m["dataset"] for m in available}
    missing_vocab = []
    for ds in datasets_with_checkpoints:
        pkl = settings.base_dir / "cache" / f"vocab_{ds}.pkl"
        if not pkl.exists():
            missing_vocab.append(str(pkl))
    if missing_vocab:
        log.warning(
            "[startup] missing vocab cache(s); first /predict for those datasets will 500. "
            "Regenerate with: python backend/generate_cache.py. Missing: %s",
            missing_vocab,
        )
    else:
        log.info("[startup] vocab caches present for all served datasets")

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="TSR-CNN-BiLSTM Prediction Service",
        description=(
            "Research-prototype API for dimensional sentiment analysis. "
            "BETA — not for clinical or diagnostic use."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predict.router)
    app.include_router(metrics.router)
    return app


app = create_app()
