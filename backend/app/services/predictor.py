"""Single-text prediction pipeline.

Parses only the user's input text with Stanza — never the training corpora.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np  # noqa: E402

from regional_cnn_lstm import (  # noqa: E402
    Config,
    DATASET_OPTIMAL_DEPTH,
    text_trees_to_regions,
    regions_to_tensor,
)
from app.services.registry import (  # noqa: E402
    DATASET_REGISTRY,
    ARCHITECTURES,
    DIMENSIONS,
    RUN_TAGS,
)
from app.services.cache import ResourceCache  # noqa: E402

log = logging.getLogger("tsr-backend.predictor")


def predict_text(
    text: str,
    dataset: str,
    dimension: str,
    architecture: str,
    run_tag: str,
    cache: ResourceCache,
) -> Dict[str, Any]:
    """Run inference on a single string.

    Pipeline: prepare_for_parser -> Stanza constituency parse -> region
    extraction at optimal depth -> tensor encode -> model.predict ->
    clamp to scale -> collect warnings.
    """
    # -- validate ---------------------------------------------------------
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"unknown dataset: {dataset!r}. known: {sorted(DATASET_REGISTRY)}"
        )
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}; got {dimension!r}")
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}; got {architecture!r}"
        )
    if run_tag not in RUN_TAGS:
        raise ValueError(f"run_tag must be one of {RUN_TAGS}; got {run_tag!r}")
    if dataset == "sst" and dimension == "arousal":
        raise ValueError("SST has no arousal labels; request valence instead.")

    depth = DATASET_OPTIMAL_DEPTH[dataset][dimension]

    # -- parse the user's input only --------------------------------------
    proc = cache.processor(dataset)
    parser = cache.parser(dataset)
    prepared = proc.prepare_for_parser(text)
    doc = parser._nlp(prepared)
    sent_trees = [sent.constituency for sent in doc.sentences]
    regions = text_trees_to_regions(sent_trees, depth)
    n_regions = len(regions)
    n_tokens_total = sum(len(r) for r in regions)

    # -- encode -----------------------------------------------------------
    vocab = cache.vocab(dataset)
    model = cache.model(dataset, dimension, architecture, run_tag)
    _, R, N = model.input_shape

    cfg_enc = Config(base_dir=cache.base_dir)
    cfg_enc.case_sensitive = False
    cfg_enc.max_num_regions = R
    cfg_enc.max_region_len = N

    x = regions_to_tensor([regions], vocab, cfg_enc)

    # Clip token IDs that exceed the model's embedding vocab.
    # The dataset vocab may be larger than the model's training-time vocab.
    # Map out-of-range IDs to UNK (or 0 if UNK itself is out of range).
    model_vocab_size = None
    for layer in model.layers:
        if hasattr(layer, "input_dim"):
            model_vocab_size = layer.input_dim
            break
    if model_vocab_size is not None:
        oor_mask = x >= model_vocab_size
        if oor_mask.any():
            unk_id = cfg_enc.unk_token_id
            if unk_id >= model_vocab_size:
                unk_id = 0
            x = np.where(oor_mask, unk_id, x)
            log.warning(
                "[predict] clipped %d token IDs >= %d to UNK=%d for %s/%s/%s",
                int(oor_mask.sum()), model_vocab_size, unk_id,
                dataset, dimension, architecture,
            )

    # -- predict ----------------------------------------------------------
    raw = model.predict(x, verbose=0)
    score_raw = float(raw.flatten()[0])

    # -- clamp to dataset scale + collect warnings ------------------------
    scale_min, scale_max = DATASET_REGISTRY[dataset]["scale"]
    clamped = float(max(scale_min, min(scale_max, score_raw)))

    log.info(
        "[predict] %s/%s/%s/%s | raw=%.4f clamped=%.4f regions=%d/%d tokens=%d",
        dataset, dimension, architecture, run_tag,
        score_raw, clamped, n_regions, R, n_tokens_total,
    )

    warnings: List[str] = []
    if n_regions > R:
        warnings.append(
            f"Text had {n_regions} regions at depth {depth}; truncated to {R}. "
            "Prediction ignores later content."
        )
    if n_regions < 2:
        warnings.append(
            f"Text produced only {n_regions} region at depth {depth}. "
            "Single-region inputs may yield low-confidence predictions."
        )
    if score_raw != clamped:
        warnings.append(
            f"Raw output {score_raw:.3f} fell outside scale [{scale_min}, {scale_max}] "
            "and was clamped. This usually indicates out-of-distribution input."
        )

    unk_id = cfg_enc.unk_token_id
    token_ids = x.flatten()
    nonzero = token_ids[token_ids != 0]
    if len(nonzero) > 0:
        oov_rate = float(np.mean(nonzero == unk_id))
        if oov_rate > 0.3:
            warnings.append(
                f"{oov_rate:.0%} of tokens were out-of-vocabulary. "
                "Prediction may be unreliable."
            )

    return {
        "score": clamped,
        "score_raw": score_raw,
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "dataset": dataset,
        "dimension": dimension,
        "architecture": architecture,
        "run_tag": run_tag,
        "region_depth": int(depth),
        "region_capacity": int(R),
        "n_regions": int(n_regions),
        "n_tokens_total": int(n_tokens_total),
        "regions": [list(r) for r in regions],
        "warnings": warnings,
    }
