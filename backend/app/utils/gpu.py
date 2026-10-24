"""GPU/CPU setup. Forces TensorFlow to CPU on low-VRAM hosts so that
PyTorch (Stanza) and TF do not contend for the same GPU memory.

.env sets CUDA_VISIBLE_DEVICES=-1 by default, which is respected here.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("tsr-backend.gpu")


def configure_gpu() -> None:
    """Set TF memory-growth and honor CUDA_VISIBLE_DEVICES."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF info logs
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        log.warning("[gpu] tensorflow import failed: %s", exc)
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        log.info("[gpu] no GPU visible — TensorFlow will use CPU")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            log.warning("[gpu] could not set memory growth on %s: %s", gpu, exc)
    log.info("[gpu] configured %d GPU(s) with memory growth", len(gpus))


def log_device_info() -> None:
    try:
        import tensorflow as tf
        log.info("[gpu] tensorflow=%s devices=%s", tf.__version__, tf.config.list_physical_devices())
    except Exception:
        pass
