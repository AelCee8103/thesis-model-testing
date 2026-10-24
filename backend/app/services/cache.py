"""Lazy cache of Stanza pipelines, vocabularies, and Keras models.

Strict design: vocabularies MUST come from `backend/cache/vocab_{dataset}.pkl`.
The service never loads a training CSV and never re-parses a training dataset
at runtime. Missing vocab => FileNotFoundError with a remediation message.

Only the user's single input text is parsed with Stanza at prediction time.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# backend root on sys.path so regional_cnn_lstm is importable
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from regional_cnn_lstm import (  # noqa: E402
    Config,
    TreeParser,
    DATASET_OPTIMAL_DEPTH,
    build_model,
    load_cache,
)
from app.services.registry import DATASET_REGISTRY  # noqa: E402

log = logging.getLogger("tsr-backend.cache")


class ResourceCache:
    """Serves pre-built artifacts; never touches training CSVs at runtime."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).resolve()
        self._cache_dir = self.base_dir / "cache"

        self._cfg = Config(base_dir=self.base_dir)
        self._cfg.case_sensitive = False
        self._cfg.cache_dir = self._cache_dir

        self._processors: Dict[str, Any] = {}
        self._parsers: Dict[str, TreeParser] = {}
        self._vocabs: Dict[str, Dict[str, int]] = {}
        self._models: Dict[tuple, Any] = {}
        self._model_lock = threading.Lock()

    # -- processors (single-text prep only; never calls .load()) ----------
    def processor(self, dataset: str) -> Any:
        if dataset not in self._processors:
            meta = DATASET_REGISTRY[dataset]
            cls = meta["processor_cls"]
            csv_path = self.base_dir / meta["csv"]
            # Instantiating the processor reads no data — it stores the path
            # and returns immediately. We only use prepare_for_parser().
            self._processors[dataset] = cls(filepath=csv_path)
        return self._processors[dataset]

    # -- Stanza pipeline (user input parsing only) ------------------------
    def parser(self, dataset: str) -> TreeParser:
        if dataset not in self._parsers:
            proc = self.processor(dataset)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            p = TreeParser(self._cache_dir, proc)
            p._ensure_pipeline()
            self._parsers[dataset] = p
        return self._parsers[dataset]

    # -- vocab (STRICT: disk cache only; never rebuild) -------------------
    def vocab(self, dataset: str) -> Dict[str, int]:
        if dataset in self._vocabs:
            return self._vocabs[dataset]

        pkl_path = self._cache_dir / f"vocab_{dataset}.pkl"
        cached = load_cache(pkl_path)
        if cached is None:
            raise FileNotFoundError(
                f"vocab cache missing: {pkl_path}. "
                "Regenerate with: python backend/generate_cache.py"
            )
        vocab = cached["vocab"] if isinstance(cached, dict) and "vocab" in cached else cached
        if not isinstance(vocab, dict):
            raise ValueError(
                f"vocab cache at {pkl_path} has unexpected format: {type(vocab).__name__}"
            )
        self._vocabs[dataset] = vocab
        log.info("[vocab] loaded %s (%d entries)", pkl_path.name, len(vocab))
        return vocab

    # -- Keras model (h5-rebuild to survive version mismatches) -----------
    def model(
        self,
        dataset: str,
        dimension: str,
        architecture: str,
        run_tag: str,
    ) -> Any:
        key = (dataset, dimension, architecture, run_tag)
        if key in self._models:
            return self._models[key]

        with self._model_lock:
            if key in self._models:
                return self._models[key]

            path = _find_checkpoint(self.base_dir, dataset, dimension, architecture, run_tag)
            if path is None or not path.exists():
                raise FileNotFoundError(
                    f"No trained model for dataset={dataset}, dimension={dimension}, "
                    f"architecture={architecture}, run_tag={run_tag}. "
                    f"Expected: {self.base_dir}/models/{dataset}/{run_tag}/{dimension}/{architecture}.keras"
                )

            log.info("[model] loading %s", path)
            model = _load_model_from_keras(path)
            self._models[key] = model
            log.info("[model] loaded; input=%s output=%s", model.input_shape, model.output_shape)
            return model


def _find_checkpoint(
    base_dir: Path,
    dataset: str,
    dimension: str,
    architecture: str,
    run_tag: str,
) -> Optional[Path]:
    """Find a checkpoint in the clean layout:
    base_dir/models/{dataset}/{run_tag}/{dimension}/{architecture}.keras
    """
    if dataset not in DATASET_REGISTRY:
        return None
    if dimension not in DATASET_OPTIMAL_DEPTH.get(dataset, {}):
        return None
    return (
        base_dir / "models" / dataset / run_tag / dimension / f"{architecture}.keras"
    )


def _load_model_from_keras(keras_path: Path) -> Any:
    """Load a .keras file by rebuilding the model from code and loading
    weights from the embedded h5. This bypasses Keras serialization, which
    breaks across TF/Keras version mismatches between training and serving.
    """
    z = zipfile.ZipFile(keras_path)
    config = json.loads(z.read("config.json"))

    tmpdir = tempfile.mkdtemp()
    weights_path = Path(tmpdir) / "model.weights.h5"
    try:
        with open(weights_path, "wb") as f:
            f.write(z.read("model.weights.h5"))

        with h5py.File(weights_path, "r") as wf:
            emb_w = wf["layers/embedding/vars/0"][()]
            conv_kernel = wf["layers/time_distributed/layer/vars/0"][()]
            conv_bias = wf["layers/time_distributed/layer/vars/1"][()]

            if "bidirectional" in wf["layers"]:
                is_bilstm = True
                fwd_kernel = wf["layers/bidirectional/forward_layer/cell/vars/0"][()]
                fwd_recurrent = wf["layers/bidirectional/forward_layer/cell/vars/1"][()]
                fwd_bias = wf["layers/bidirectional/forward_layer/cell/vars/2"][()]
                bwd_kernel = wf["layers/bidirectional/backward_layer/cell/vars/0"][()]
                bwd_recurrent = wf["layers/bidirectional/backward_layer/cell/vars/1"][()]
                bwd_bias = wf["layers/bidirectional/backward_layer/cell/vars/2"][()]
                lstm_units = fwd_kernel.shape[1] // 4  # 4 gates
            else:
                is_bilstm = False
                lstm_kernel = wf["layers/lstm/cell/vars/0"][()]
                lstm_recurrent = wf["layers/lstm/cell/vars/1"][()]
                lstm_bias = wf["layers/lstm/cell/vars/2"][()]
                lstm_units = lstm_kernel.shape[1] // 4

            dense_kernel = wf["layers/dense/vars/0"][()]
            dense_bias = wf["layers/dense/vars/1"][()]

        # Parse input shape and CNN hyperparams from the embedded config
        layers_cfg = config["config"]["layers"]
        input_shape = None
        conv_filters = None
        conv_kernel_size = None
        pool_size = None
        spatial_dropout_rate = None
        embedding_dim = emb_w.shape[1]

        for layer in layers_cfg:
            cls = layer["class_name"]
            cfg = layer.get("config", {})
            if cls == "InputLayer":
                bs = cfg.get("batch_input_shape")
                if bs and len(bs) == 3:
                    input_shape = (bs[1], bs[2])
            elif cls == "TimeDistributed":
                inner = cfg.get("layer", {})
                inner_cls = inner.get("class_name")
                inner_cfg = inner.get("config", {})
                if inner_cls == "Conv1D":
                    conv_filters = inner_cfg.get("filters")
                    ks = inner_cfg.get("kernel_size")
                    conv_kernel_size = ks[0] if isinstance(ks, list) else ks
                elif inner_cls == "MaxPooling1D":
                    ps = inner_cfg.get("pool_size")
                    pool_size = ps[0] if isinstance(ps, list) else ps
                elif inner_cls == "SpatialDropout1D":
                    spatial_dropout_rate = inner_cfg.get("rate")

        if input_shape is None:
            raise ValueError(f"could not parse model config from {keras_path}")

        R, N = input_shape
        vocab_size = emb_w.shape[0]
        output_units = dense_kernel.shape[1]

        # Unique prefix so Keras 3 variable names don't clash across multiple loads
        prefix = f"{keras_path.parent.parent.name}_{keras_path.parent.name}_{keras_path.stem}"

        log.info(
            "[model] rebuilding: vocab=%d input=(%d,%d) lstm_units=%d bilstm=%s "
            "output=%d name=%s",
            vocab_size, R, N, lstm_units, is_bilstm, output_units, keras_path.name,
        )

        cfg = Config(base_dir=Path(__file__).resolve().parents[2])
        cfg.case_sensitive = False
        cfg.max_num_regions = R
        cfg.max_region_len = N
        cfg.target_dimension = "valence" if output_units == 1 else None
        cfg.bidirectional_lstm = is_bilstm
        cfg.lstm_units = lstm_units
        cfg.trainable_embeddings = False
        cfg.cnn_filters = conv_filters or 60
        cfg.cnn_kernel_size = conv_kernel_size or 3
        cfg.pool_size = pool_size or 2
        cfg.spatial_dropout = spatial_dropout_rate or 0.25
        cfg.embedding_dropout = 0.0
        cfg.post_lstm_dropout = 0.0
        cfg.l2_reg = 0.0
        cfg.learning_rate = 1e-3

        rng = np.random.RandomState(42)
        emb_matrix = rng.normal(0, 0.01, (vocab_size, embedding_dim)).astype(np.float32)

        model = build_model(vocab_size, emb_matrix, cfg, name_prefix=prefix)

        # Load weights — names prefixed, shapes guaranteed to match
        model.get_layer(f"{prefix}_glove_embedding").set_weights([emb_w])
        model.get_layer(f"{prefix}_td_conv").layer.set_weights([conv_kernel, conv_bias])

        if is_bilstm:
            model.get_layer(f"{prefix}_region_bilstm").forward_layer.set_weights(
                [fwd_kernel, fwd_recurrent, fwd_bias]
            )
            model.get_layer(f"{prefix}_region_bilstm").backward_layer.set_weights(
                [bwd_kernel, bwd_recurrent, bwd_bias]
            )
        else:
            model.get_layer(f"{prefix}_region_lstm").set_weights(
                [lstm_kernel, lstm_recurrent, lstm_bias]
            )

        decoder_tag = "v_linear_decoder" if output_units == 1 else "va_linear_decoder"
        model.get_layer(f"{prefix}_{decoder_tag}").set_weights([dense_kernel, dense_bias])

        return model
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
