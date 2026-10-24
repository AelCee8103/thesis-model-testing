#!/usr/bin/env python3
"""Inference-speed benchmark for the saved Regional CNN-LSTM models.

Speed-only. Does NOT retrain, does NOT modify any model or data file.
Loads each saved checkpoint (baseline `lstm` + proposed `bilstm`) and times
`model.predict()` on the full row set of every dataset, for both valence and
arousal. Reports latency (ms/sample) and throughput (samples/sec) for the
thesis Results & Discussion chapter.

Same rows feed every model -> identical workload -> fair lstm-vs-bilstm timing.
No train/test split: speed depends only on sample count and input shape, not on
whether a row was trained on.

Usage:
    cd backend
    python benchmark_speed.py                       # all datasets
    python benchmark_speed.py --dataset emobank_writer
    python benchmark_speed.py --runs 5 --batch-size 32
"""
from __future__ import annotations

# --- device + quiet logs MUST be decided BEFORE importing tensorflow --------
import os
import sys as _sys

# GPU is opt-in via --gpu. Default = CPU for stable, reproducible timing.
# This must run before tensorflow is imported, hence the early argv peek.
if "--gpu" not in _sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import csv as csvmod
import glob
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np  # noqa: E402

from regional_cnn_lstm import (  # noqa: E402
    Config,
    EmoBankWriterProcessor,
    EmoBankReaderProcessor,
    FBProcessor,
    CVATProcessor,
    SSTProcessor,
    DATASET_OPTIMAL_DEPTH,
    text_trees_to_regions,
    regions_to_tensor,
    load_cache,
)
from app.services.cache import ResourceCache  # noqa: E402

# dataset -> (processor class, expected csv basename)
DATASETS: Dict[str, tuple] = {
    "emobank_writer": (EmoBankWriterProcessor, "writer_10240.csv"),
    "emobank_reader": (EmoBankReaderProcessor, "reader_10240.csv"),
    "fb": (FBProcessor, "fb_dataset_cleaned.csv"),
    "cvat": (CVATProcessor, "CVAT.csv"),
    "sst": (SSTProcessor, "stanford_sentiment_assembled.csv"),
}

ARCHITECTURES = ["lstm", "bilstm"]
RUN_TAG = "paper"
DATASETS_DIR = _BACKEND_DIR / "datasets"
CACHE_DIR = _BACKEND_DIR / "cache"
RESULTS_DIR = _BACKEND_DIR / "results"


def resolve_csv(basename: str) -> Optional[Path]:
    """Find the dataset CSV, tolerating ' (1)'/' (2)' download suffixes."""
    exact = DATASETS_DIR / basename
    if exact.exists():
        return exact
    stem = Path(basename).stem
    matches = sorted(glob.glob(str(DATASETS_DIR / f"{stem}*.csv")))
    return Path(matches[0]) if matches else None


def model_vocab_size(model) -> Optional[int]:
    for layer in model.layers:
        if hasattr(layer, "input_dim"):
            return layer.input_dim
    return None


def encode(regions: List[List[List[str]]], vocab: Dict[str, int], R: int, N: int) -> np.ndarray:
    """Region words -> padded int tensor (n, R, N), clipping OOR ids to UNK.

    Mirrors the runtime predictor: the cached vocab is full-data and may be
    larger than the model's training-time embedding, so ids >= embedding size
    are mapped to UNK (predictor.py:89-108).
    """
    cfg = Config(base_dir=_BACKEND_DIR)
    cfg.case_sensitive = False
    cfg.max_num_regions = R
    cfg.max_region_len = N
    return regions_to_tensor(regions, vocab, cfg)


def clip_oor(x: np.ndarray, vocab_size: Optional[int]) -> np.ndarray:
    if vocab_size is None:
        return x
    oor = x >= vocab_size
    if oor.any():
        unk = 1 if 1 < vocab_size else 0
        x = np.where(oor, unk, x)
    return x


def time_predict(model, x: np.ndarray, runs: int, batch_size: int) -> tuple:
    """Warm-up (discarded) then `runs` timed model.predict calls.

    Returns (mean_total_s, std_total_s, totals).
    """
    model.predict(x[:1], batch_size=1, verbose=0)  # warm-up: build graph
    totals: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x, batch_size=batch_size, verbose=0)
        totals.append(time.perf_counter() - t0)
    arr = np.asarray(totals, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), totals


def dims_for(dataset: str) -> List[str]:
    return list(DATASET_OPTIMAL_DEPTH.get(dataset, {}).keys())


def sample_indices(n: int, frac: float, seed: int = 42) -> np.ndarray:
    """Seed-42 random subset of row indices (sorted).

    Depends only on (n, frac, seed) -> identical rows for every model of a
    dataset, so lstm/bilstm and valence/arousal all time the same workload.
    frac >= 1.0 returns every row.
    """
    if frac >= 1.0:
        return np.arange(n)
    k = max(1, int(round(n * frac)))
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def env_header(threads: int, batch_size: int, runs: int, sample: float) -> Dict[str, str]:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = []
        for g in gpus:
            try:
                d = tf.config.experimental.get_device_details(g)
                names.append(d.get("device_name", g.name))
            except Exception:
                names.append(g.name)
        device = f"GPU x{len(gpus)} ({', '.join(names)})"
    else:
        device = "CPU (CUDA_VISIBLE_DEVICES=-1)"

    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    ram_gb = "unknown"
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        ram_gb = f"{pages * page / 1e9:.1f}"
    except (ValueError, OSError):
        pass
    return {
        "CPU": cpu_model,
        "Cores (logical)": str(os.cpu_count()),
        "RAM (GB)": ram_gb,
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "TensorFlow": tf.__version__,
        "Device": device,
        "Intra-op threads": str(threads),
        "Batch size": str(batch_size),
        "Timed runs": str(runs),
        "Sample fraction": ("full (100%)" if sample >= 1.0 else f"{sample:.0%} (seed-42 random)"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference-speed benchmark (no retrain)")
    ap.add_argument("--dataset", choices=list(DATASETS), help="single dataset only")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per model (default 5)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None,
                    help="pin TF intra-op threads for reproducible CPU timing")
    ap.add_argument("--gpu", action="store_true",
                    help="use CUDA GPU instead of CPU (handled before TF import)")
    ap.add_argument("--sample", type=float, default=0.1,
                    help="fraction of rows to time, seed-42 random "
                         "(default 0.1 = 10%%; use 1.0 for the full set)")
    args = ap.parse_args()

    import tensorflow as tf

    if args.gpu and not tf.config.list_physical_devices("GPU"):
        print("WARNING: --gpu requested but TensorFlow sees no GPU. "
              "Check CUDA/cuDNN install (nvidia-smi, tf build). Falling back to CPU.")
    if args.threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    threads = tf.config.threading.get_intra_op_parallelism_threads()

    cache = ResourceCache(_BACKEND_DIR)
    targets = [args.dataset] if args.dataset else list(DATASETS)
    rows: List[Dict] = []

    env = env_header(threads, args.batch_size, args.runs, args.sample)
    print("=" * 72)
    for k, v in env.items():
        print(f"  {k:18s}: {v}")
    print("=" * 72)

    for ds in targets:
        proc_cls, csv_name = DATASETS[ds]
        csv_path = resolve_csv(csv_name)
        if csv_path is None:
            print(f"[{ds}] SKIP: csv not found (looked for {csv_name} in {DATASETS_DIR})")
            continue

        trees_cache = load_cache(CACHE_DIR / f"parsed_trees_{ds}.pkl")
        if not (isinstance(trees_cache, dict) and "trees" in trees_cache):
            print(f"[{ds}] SKIP: parsed_trees cache missing/old format")
            continue
        trees = trees_cache["trees"]

        proc = proc_cls(filepath=csv_path)
        df = proc.load()
        if len(trees) != len(df):
            print(f"[{ds}] SKIP: tree/row mismatch (trees={len(trees)} rows={len(df)})")
            continue
        n_total = len(df)
        sample_idx = sample_indices(n_total, args.sample)
        sel_trees = [trees[i] for i in sample_idx]
        n = len(sel_trees)
        vocab = cache.vocab(ds)

        # --- validate the sample really is the requested fraction ------------
        expected = n_total if args.sample >= 1.0 else max(1, int(round(args.sample * n_total)))
        actual_frac = n / n_total
        assert n == expected, (
            f"[{ds}] sample size {n} != expected {expected} "
            f"({args.sample:.0%} of {n_total})"
        )
        assert len(set(sample_idx.tolist())) == n, f"[{ds}] sampled rows contain duplicates"
        assert sample_idx.min() >= 0 and sample_idx.max() < n_total, \
            f"[{ds}] sampled index out of range"

        frac_lbl = "full" if args.sample >= 1.0 else f"{args.sample:.0%}"
        print(f"\n[{ds}] ✓ sampled {n} of {n_total} rows = {actual_frac:.1%} "
              f"(requested {frac_lbl}) | csv={csv_path.name}")
        for dim in dims_for(ds):
            depth = DATASET_OPTIMAL_DEPTH[ds][dim]
            regions = [text_trees_to_regions(t, depth) for t in sel_trees]
            for arch in ARCHITECTURES:
                model = cache.model(ds, dim, arch, RUN_TAG)
                _, R, N = model.input_shape
                x = clip_oor(encode(regions, vocab, R, N), model_vocab_size(model))
                mean_s, std_s, _ = time_predict(model, x, args.runs, args.batch_size)
                latency_ms = mean_s / n * 1000.0
                throughput = n / mean_s
                rows.append({
                    "dataset": ds,
                    "dimension": dim,
                    "architecture": arch,
                    "n_samples": n,
                    "latency_ms_per_sample": round(latency_ms, 4),
                    "throughput_per_sec": round(throughput, 2),
                    "total_time_mean_s": round(mean_s, 4),
                    "total_time_std_s": round(std_s, 4),
                })
                print(f"    {dim:8s} {arch:7s} | "
                      f"{latency_ms:8.3f} ms/sample | {throughput:9.1f} /s | "
                      f"total {mean_s:.3f}±{std_s:.3f}s")

    if not rows:
        print("\nNo results produced.")
        return

    # average BiLSTM/LSTM slowdown across all dataset×dimension pairs
    pair: Dict[tuple, Dict[str, float]] = {}
    for r in rows:
        pair.setdefault((r["dataset"], r["dimension"]), {})[r["architecture"]] = \
            r["latency_ms_per_sample"]
    ratios = [v["bilstm"] / v["lstm"] for v in pair.values()
              if v.get("lstm") and v.get("bilstm")]
    print("\n" + "-" * 72)
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"  SUMMARY: proposed BiLSTM is on average {avg:.2f}× "
              f"({(avg - 1) * 100:+.0f}%) slower than baseline LSTM.")
    print("  Lower latency = faster | Higher throughput = faster")
    print("-" * 72)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    write_report(rows, env)
    print(f"\nWrote:\n  {RESULTS_DIR / 'speed_results.csv'}"
          f"\n  {RESULTS_DIR / 'speed_report.md'}")


def write_csv(rows: List[Dict]) -> None:
    path = RESULTS_DIR / "speed_results.csv"
    fields = ["dataset", "dimension", "architecture", "n_samples",
              "latency_ms_per_sample", "throughput_per_sec",
              "total_time_mean_s", "total_time_std_s"]
    with open(path, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


DATASET_LABELS = {
    "emobank_writer": "EmoBank (writer)",
    "emobank_reader": "EmoBank (reader)",
    "fb": "Facebook",
    "cvat": "CVAT (Chinese)",
    "sst": "SST",
}


def write_report(rows: List[Dict], env: Dict[str, str]) -> None:
    # group: (dataset, dimension) -> {arch: row}
    by_key: Dict[tuple, Dict[str, Dict]] = {}
    for r in rows:
        by_key.setdefault((r["dataset"], r["dimension"]), {})[r["architecture"]] = r

    # collect pairwise slowdowns for the summary
    slowdowns: List[float] = []
    for archs in by_key.values():
        lo, hi = archs.get("lstm"), archs.get("bilstm")
        if lo and hi and lo["latency_ms_per_sample"]:
            slowdowns.append(hi["latency_ms_per_sample"] / lo["latency_ms_per_sample"])
    avg_ratio = sum(slowdowns) / len(slowdowns) if slowdowns else 0.0

    sample_lbl = env.get("Sample fraction", "")
    is_sampled = not sample_lbl.startswith("full")

    L: List[str] = []
    L.append("# Inference Speed: Baseline (LSTM) vs Proposed (BiLSTM)\n")

    # ---- TL;DR --------------------------------------------------------------
    L.append("## Summary\n")
    if avg_ratio:
        pct = (avg_ratio - 1) * 100
        faster_slower = "slower" if avg_ratio >= 1 else "faster"
        L.append(f"- On average the **proposed BiLSTM is {avg_ratio:.2f}× "
                 f"({abs(pct):.0f}% {faster_slower})** than the baseline LSTM at inference.")
    L.append(f"- Device: **{env.get('Device', '?')}**, batch size {env.get('Batch size', '?')}, "
             f"averaged over {env.get('Timed runs', '?')} runs.")
    L.append(f"- Measured on **{sample_lbl}** of each dataset (same rows for both models).")
    L.append("- **Lower latency = faster. Higher throughput = faster.**")
    if is_sampled:
        L.append("- ⚠️ This is a *sampled* run for speed — absolute numbers are noisy. "
                 "For final thesis figures re-run with `--sample 1.0` (full data).")
    L.append("")

    # ---- how to read --------------------------------------------------------
    L.append("## How to read the table\n")
    L.append("| Term | Meaning |")
    L.append("|---|---|")
    L.append("| **Latency (ms/sample)** | Time to score one text, in milliseconds. Lower is better. |")
    L.append("| **Throughput (texts/s)** | How many texts scored per second. Higher is better. |")
    L.append("| **BiLSTM slower by** | How much slower the proposed model is vs the baseline. |")
    L.append("| **N** | Number of texts timed (the sampled rows). |")
    L.append("")

    # ---- main side-by-side table -------------------------------------------
    L.append("## Results — LSTM vs BiLSTM (side by side)\n")
    L.append("| Dataset | Dimension | N | LSTM latency (ms) | BiLSTM latency (ms) | "
             "BiLSTM slower by | LSTM throughput (/s) | BiLSTM throughput (/s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (ds, dim), archs in by_key.items():
        lo, hi = archs.get("lstm"), archs.get("bilstm")
        name = DATASET_LABELS.get(ds, ds)
        n = (lo or hi)["n_samples"]
        lo_lat = f"{lo['latency_ms_per_sample']:.3f}" if lo else "—"
        hi_lat = f"{hi['latency_ms_per_sample']:.3f}" if hi else "—"
        lo_tp = f"{lo['throughput_per_sec']:.0f}" if lo else "—"
        hi_tp = f"{hi['throughput_per_sec']:.0f}" if hi else "—"
        slower = "—"
        if lo and hi and lo["latency_ms_per_sample"]:
            r = hi["latency_ms_per_sample"] / lo["latency_ms_per_sample"]
            slower = f"{r:.2f}× ({(r - 1) * 100:+.0f}%)"
        L.append(f"| {name} | {dim} | {n} | {lo_lat} | {hi_lat} | {slower} | {lo_tp} | {hi_tp} |")
    L.append("")
    L.append("> *Slower by* > 1× means the proposed BiLSTM takes longer than the baseline — "
             "expected, because BiLSTM reads each text both forward and backward. This is the "
             "**accuracy-vs-speed tradeoff**: pair this table with the accuracy results to argue "
             "the BiLSTM's quality gain is worth the modest speed cost.\n")

    # ---- environment (moved to the end as reference) ------------------------
    L.append("## Test environment\n")
    L.append("| Setting | Value |")
    L.append("|---|---|")
    for k, v in env.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("_Raw per-model numbers (incl. run-to-run std): `speed_results.csv`._\n")

    (RESULTS_DIR / "speed_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
