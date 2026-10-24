#!/usr/bin/env python3
"""Regenerate all parse-tree and vocab cache files for the prediction service.

This is a one-time script to be run locally. It:
  1. Downloads Stanza models (one-time, ~800MB)
  2. Parses every dataset and saves parsed_trees_{dataset}.pkl
  3. Builds vocab from the parsed trees and saves vocab_{dataset}.pkl
  4. Computes label statistics and saves label_stats_{dataset}_{dimension}.pkl

After this, the FastAPI prediction service will start in seconds
because all expensive computations are cached to disk.

Usage:
    cd backend
    source ../.venv/bin/activate
    python generate_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from regional_cnn_lstm import (
    Config,
    EmoBankWriterProcessor,
    EmoBankReaderProcessor,
    FBProcessor,
    CVATProcessor,
    SSTProcessor,
    TreeParser,
    DATASET_OPTIMAL_DEPTH,
    build_vocab,
    text_trees_to_regions,
    load_cache,
    save_cache,
)

import numpy as np

DATASETS = {
    "emobank_writer": EmoBankWriterProcessor,
    "emobank_reader": EmoBankReaderProcessor,
    "fb": FBProcessor,
    "cvat": CVATProcessor,
    "sst": SSTProcessor,
}

CSV_FILES = {
    "emobank_writer": "writer_10240.csv",
    "emobank_reader": "reader_10240.csv",
    "fb": "fb_dataset_cleaned.csv",
    "cvat": "CVAT.csv",
    "sst": "stanford_sentiment_assembled.csv",
}

EMBEDDING_FILES = {
    "emobank_writer": "glove.840B.300d.txt",
    "emobank_reader": "glove.840B.300d.txt",
    "fb": "glove.840B.300d.txt",
    "cvat": "sgns.merge.word",
    "sst": "glove.840B.300d.txt",
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cache files for all datasets")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Project base directory containing CSVs and embeddings")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Regenerate cache for a single dataset only")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent

    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(base_dir=base_dir)
    cfg.cache_dir = cache_dir
    cfg.case_sensitive = False

    datasets_to_process = list(DATASETS.keys())
    if args.dataset:
        datasets_to_process = [args.dataset]

    print(f"Base directory: {base_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Datasets to process: {datasets_to_process}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Parse trees
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 1: Generating parse-tree caches")
    print("=" * 60)

    for dataset_name in datasets_to_process:
        pkl_path = cache_dir / f"parsed_trees_{dataset_name}.pkl"
        existing = load_cache(pkl_path)
        if existing is not None:
            if isinstance(existing, dict) and "trees" in existing:
                first_tree = existing["trees"][0][0] if existing["trees"] and existing["trees"][0] else None
                if first_tree is not None and hasattr(first_tree, "children"):
                    print(f"  [{dataset_name}] parse-tree cache already valid ({pkl_path}), skipping")
                    continue
            print(f"  [{dataset_name}] existing cache is old format, regenerating...")

        proc_cls = DATASETS[dataset_name]
        csv_path = base_dir / CSV_FILES[dataset_name]
        proc = proc_cls(filepath=csv_path)

        print(f"  [{dataset_name}] Loading dataset...")
        df = proc.load()
        texts = proc.get_texts(df)
        print(f"  [{dataset_name}] {len(texts)} texts loaded")

        print(f"  [{dataset_name}] Parsing with Stanza (lang={proc.language})...")
        t0 = time.time()
        tree_parser = TreeParser(cache_dir, proc)
        trees = tree_parser.parse_all(texts)
        elapsed = time.time() - t0
        print(f"  [{dataset_name}] Parsed in {elapsed:.1f}s ({len(trees)} texts)")
        print()

    # ------------------------------------------------------------------
    # Phase 2: Vocabularies
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 2: Generating vocabulary caches")
    print("=" * 60)

    for dataset_name in datasets_to_process:
        pkl_path = cache_dir / f"vocab_{dataset_name}.pkl"
        existing = load_cache(pkl_path)
        if existing is not None and isinstance(existing, dict) and "vocab" in existing:
            n_entries = len(existing["vocab"])
            print(f"  [{dataset_name}] vocab cache already valid ({n_entries} entries), skipping")
            continue

        # Load parse trees from cache (should be valid now)
        ptf_path = cache_dir / f"parsed_trees_{dataset_name}.pkl"
        cached_trees = load_cache(ptf_path)
        if cached_trees is None:
            print(f"  [{dataset_name}] ERROR: no parse trees found at {ptf_path}")
            continue

        if isinstance(cached_trees, dict) and "trees" in cached_trees:
            trees = cached_trees["trees"]
        elif isinstance(cached_trees, list):
            print(f"  [{dataset_name}] ERROR: parse trees still in old format, cannot build vocab")
            continue
        else:
            print(f"  [{dataset_name}] ERROR: unexpected cache format: {type(cached_trees)}")
            continue

        dims = DATASET_OPTIMAL_DEPTH.get(dataset_name, {"valence": 5})
        target_depth = max(dims.values())
        all_text_regions = [text_trees_to_regions(t, target_depth) for t in trees]

        vocab = build_vocab(all_text_regions, cfg)
        save_cache({"vocab": vocab}, pkl_path)
        print(f"  [{dataset_name}] vocab: {len(vocab)} entries saved to {pkl_path}")

    # ------------------------------------------------------------------
    # Phase 3: Label statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PHASE 3: Generating label-statistics caches")
    print("=" * 60)

    for dataset_name in datasets_to_process:
        proc_cls = DATASETS[dataset_name]
        csv_path = base_dir / CSV_FILES[dataset_name]
        proc = proc_cls(filepath=csv_path)

        dimensions = ["valence"]
        if dataset_name != "sst":
            dimensions.append("arousal")

        for dim in dimensions:
            pkl_path = cache_dir / f"label_stats_{dataset_name}_{dim}.pkl"
            existing = load_cache(pkl_path)
            if existing is not None and isinstance(existing, dict) and "mean" in existing:
                print(f"  [{dataset_name}/{dim}] label_stats cache already valid, skipping")
                continue

            df = proc.load()
            labels = proc.get_labels(df)
            dim_idx = 0 if dim == "valence" else 1
            col = labels[:, dim_idx]
            mask = ~np.isnan(col)
            stats = {
                "mean": float(np.mean(col[mask])) if mask.any() else 0.0,
                "std": float(np.std(col[mask])) if mask.any() else 1.0,
            }
            save_cache(stats, pkl_path)
            print(f"  [{dataset_name}/{dim}] mean={stats['mean']:.4f}, std={stats['std']:.4f} saved to {pkl_path}")

    # ------------------------------------------------------------------
    # Phase 4: Embedding matrices
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PHASE 4: Generating embedding-matrix caches")
    print("=" * 60)

    for dataset_name in datasets_to_process:
        emb_path = base_dir / EMBEDDING_FILES[dataset_name]
        pkl_path = cache_dir / f"vocab_{dataset_name}.pkl"

        cached = load_cache(pkl_path)
        if cached is None or not isinstance(cached, dict) or "vocab" not in cached:
            print(f"  [{dataset_name}] ERROR: no vocab found, skipping embeddings")
            continue

        vocab = cached["vocab"]

        # Check if embedding matrix already exists
        import hashlib
        vocab_fp = hashlib.sha256("\n".join(sorted(vocab.keys())).encode("utf-8")).hexdigest()[:12]
        emb_fp = hashlib.sha256(str(emb_path.resolve()).encode("utf-8")).hexdigest()[:8]
        emb_cache = cache_dir / f"embedding_matrix_{dataset_name}_ci_{emb_fp}_{vocab_fp}.pkl"

        if emb_cache.exists():
            print(f"  [{dataset_name}] embedding matrix already cached ({emb_cache.name}), skipping")
            continue

        if not emb_path.exists():
            print(f"  [{dataset_name}] embedding file not found: {emb_path}")
            print(f"  [{dataset_name}] Skipping embedding matrix generation (predictions will use random OOV vectors)")
            continue

        print(f"  [{dataset_name}] Loading embeddings from {emb_path.name}...")
        t0 = time.time()
        matrix = load_glove_matrix(vocab, cfg, dataset_name, emb_path)
        print(f"  [{dataset_name}] Embedding matrix: {matrix.shape}, done in {time.time() - t0:.1f}s")

    print()
    print("=" * 60)
    print("ALL DONE! Cache files generated in:")
    print(f"  {cache_dir}")
    print()
    print("The FastAPI prediction service will now start in seconds")
    print("instead of minutes because all expensive computations are cached.")
    print("=" * 60)


if __name__ == "__main__":
    main()