#!/usr/bin/env python3
"""Copy trained .keras files from MODEL/ into a clean backend/models/ layout.

Source layout (Google Drive export, messy):
    MODEL/{dataset-TIMESTAMP}/{dataset-name}/regional_cnn_{arch}/{file}.keras
    MODEL/sst-TIMESTAMP/sst/sst_val_{arch}.keras        (no regional_cnn/ subdir)

Destination layout (what backend/app expects):
    backend/models/{dataset}/{run_tag}/{dimension}/{architecture}.keras

Handles:
    - Timestamped parent dirs via glob
    - Spaces in folder names ('emobank (reader)', 'emobank (writer)')
    - Dataset key remap: folder 'facebook' -> key 'fb'
    - Case variants in filenames ('facebook' vs 'FaceBook')
    - Typo tolerance ('emobang_reader_aro_lstm.keras')
    - SST flat layout (no regional_cnn_ subdir, valence-only)
    - Idempotent: skips if destination exists
    - Non-destructive: MODEL/ is untouched (uses shutil.copy2)

Run:
    python backend/scripts/reorganize_models.py
    python backend/scripts/reorganize_models.py --source ../MODEL --dest ./models
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reorganize")

# Folder-name (in MODEL/) → dataset key (in API) mapping.
# Value is the canonical key that the backend uses.
FOLDER_TO_KEY = {
    "emobank (reader)": "emobank_reader",
    "emobank (writer)": "emobank_writer",
    "facebook": "fb",
    "sst": "sst",
}

RUN_TAG = "paper"  # all current checkpoints come from the paper run

# Filename token → dimension mapping (case-insensitive).
DIM_TOKENS = {"val": "valence", "aro": "arousal"}

# Filename token → architecture mapping (case-insensitive, longest first).
ARCH_TOKENS = ("bilstm", "lstm")


def _parse_filename(path: Path) -> Optional[Tuple[str, str]]:
    """Extract (dimension, architecture) from a .keras filename.

    Accepts any underscore-delimited name containing a val/aro token and
    an lstm/bilstm token, case-insensitive. Tolerates typos in the stem
    (e.g. 'emobang_reader_aro_lstm') because we only match on the tokens.
    """
    stem = path.stem.lower()
    tokens = stem.split("_")

    dim = next((DIM_TOKENS[t] for t in tokens if t in DIM_TOKENS), None)
    arch = next((a for a in ARCH_TOKENS if a in tokens), None)
    if dim is None or arch is None:
        return None
    return dim, arch


def _discover_sources(source_root: Path) -> List[Tuple[Path, Path]]:
    """Walk MODEL/ and return a list of (src, dst_relative) pairs."""
    pairs: List[Tuple[Path, Path]] = []

    # Each timestamped dir contains exactly one dataset folder.
    for folder_name, dataset_key in FOLDER_TO_KEY.items():
        # Glob across timestamp variants: 'emobank (reader)-*'
        for timestamped in source_root.glob(f"{folder_name}-*"):
            if not timestamped.is_dir():
                continue
            # Inside each timestamped dir, the dataset folder uses the same
            # folder name WITHOUT the timestamp suffix.
            dataset_dir = timestamped / folder_name
            if not dataset_dir.is_dir():
                log.warning(
                    "expected dataset folder %r under %s — skipping",
                    folder_name, timestamped,
                )
                continue

            # .keras files live either directly under dataset_dir (sst case)
            # or under dataset_dir/regional_cnn_{lstm,bilstm}/.
            for keras_file in dataset_dir.rglob("*.keras"):
                parsed = _parse_filename(keras_file)
                if parsed is None:
                    log.warning("could not parse %s — skipping", keras_file)
                    continue
                dim, arch = parsed
                dst_rel = Path(dataset_key) / RUN_TAG / dim / f"{arch}.keras"
                pairs.append((keras_file, dst_rel))

    return pairs


def _clean_legacy_dirs(dest_root: Path) -> None:
    """Remove any old placeholder dirs (e.g. emobank-reader/) that used a
    different naming convention. Only removes empty dirs so nothing real
    is lost."""
    if not dest_root.exists():
        return
    for child in dest_root.iterdir():
        if child.is_dir() and "-" in child.name and child.name not in FOLDER_TO_KEY.values():
            # Only remove if empty tree
            has_files = any(child.rglob("*"))
            if not has_files:
                shutil.rmtree(child)
                log.info("removed empty legacy dir: %s", child)


def reorganize(source_root: Path, dest_root: Path) -> Tuple[int, int]:
    """Copy all discovered .keras files into the clean layout.

    Returns (copied, skipped) counts.
    """
    if not source_root.is_dir():
        log.error("source directory not found: %s", source_root)
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)
    _clean_legacy_dirs(dest_root)

    pairs = _discover_sources(source_root)
    if not pairs:
        log.error("no .keras files discovered under %s", source_root)
        sys.exit(1)

    copied = 0
    skipped = 0
    log.info("%-40s -> %s", "SOURCE", "DEST")
    log.info("-" * 100)
    for src, dst_rel in pairs:
        dst = dest_root / dst_rel
        if dst.exists():
            log.info("SKIP (exists)  %-38s -> %s", src.name, dst.relative_to(dest_root))
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info("COPIED         %-38s -> %s", src.name, dst.relative_to(dest_root))
        copied += 1

    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "MODEL",
        help="Source MODEL/ directory (default: project_root/MODEL)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Destination models/ directory (default: backend/models)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    log.info("source = %s", source)
    log.info("dest   = %s", dest)

    copied, skipped = reorganize(source, dest)
    log.info("-" * 100)
    log.info("done: %d copied, %d skipped (already present)", copied, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
