"""Evaluation metrics comparing baseline vs proposed predictions against truth.

Formulas mirror the thesis training pipeline so the API matches reported numbers:
`mae` / `pearson_r` reimplement regional_cnn_lstm.py:1380 / :1372, and the
Mann-Whitney U test mirrors regional_cnn_lstm.py:2110-2150.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # np.corrcoef is undefined when either side has zero variance.
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_evaluation_metrics(
    truth: list,
    baseline_pred: list,
    proposed_pred: list,
    dimension: str,
) -> Dict[str, Any]:
    if not (len(truth) == len(baseline_pred) == len(proposed_pred)):
        raise ValueError(
            "truth, baseline_pred and proposed_pred must have equal length "
            f"(got {len(truth)}, {len(baseline_pred)}, {len(proposed_pred)})."
        )
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty set.")

    y_true = np.asarray(truth, dtype=np.float64).flatten()
    y_base = np.asarray(baseline_pred, dtype=np.float64).flatten()
    y_prop = np.asarray(proposed_pred, dtype=np.float64).flatten()

    errors_baseline = np.abs(y_true - y_base)
    errors_proposed = np.abs(y_true - y_prop)

    mwu = stats.mannwhitneyu(errors_baseline, errors_proposed, alternative="two-sided")
    p_value = float(mwu.pvalue)

    return {
        "n": int(y_true.size),
        "dimension": dimension,
        "baseline": {"mae": _mae(y_true, y_base), "pearson_r": _pearson_r(y_true, y_base)},
        "proposed": {"mae": _mae(y_true, y_prop), "pearson_r": _pearson_r(y_true, y_prop)},
        "mann_whitney_u": {
            "u_statistic": float(mwu.statistic),
            "p_value": p_value,
            "significant": p_value < 0.05,
        },
    }
