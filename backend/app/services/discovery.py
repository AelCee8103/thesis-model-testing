"""Enumerate trained checkpoints on disk for the GET /models endpoint."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from regional_cnn_lstm import DATASET_OPTIMAL_DEPTH  # noqa: E402
from app.services.registry import (  # noqa: E402
    DATASET_REGISTRY,
    ARCHITECTURES,
    DIMENSIONS,
    RUN_TAGS,
)
from app.services.cache import _find_checkpoint  # noqa: E402


def discover_models(base_dir: Path) -> List[Dict[str, Any]]:
    """Return one entry per (dataset, dimension, architecture, run_tag) that has
    a checkpoint on disk. SST-arousal is skipped (no arousal labels for SST)."""
    out: List[Dict[str, Any]] = []
    for dataset, meta in DATASET_REGISTRY.items():
        dims_available = DATASET_OPTIMAL_DEPTH.get(dataset, {})
        for dimension in DIMENSIONS:
            if dimension not in dims_available:
                continue
            if dataset == "sst" and dimension == "arousal":
                continue
            for architecture in ARCHITECTURES:
                for run_tag in RUN_TAGS:
                    path = _find_checkpoint(
                        base_dir, dataset, dimension, architecture, run_tag
                    )
                    if path is None or not path.exists():
                        continue
                    out.append({
                        "dataset": dataset,
                        "dataset_description": meta["description"],
                        "language": meta["language"],
                        "dimension": dimension,
                        "architecture": architecture,
                        "run_tag": run_tag,
                        "scale_min": float(meta["scale"][0]),
                        "scale_max": float(meta["scale"][1]),
                        "region_depth": int(dims_available[dimension]),
                        "checkpoint_exists": True,
                    })
    return out
