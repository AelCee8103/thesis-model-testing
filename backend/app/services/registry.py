"""Single source of truth for the datasets exposed by the API."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Make regional_cnn_lstm importable (lives at backend root, one level above app/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from regional_cnn_lstm import (  # noqa: E402
    EmoBankWriterProcessor,
    EmoBankReaderProcessor,
    FBProcessor,
    SSTProcessor,
    CVATProcessor,
)

# Dataset key → (processor class, csv, scale, language, description).
# CVAT entries are listed; checkpoints are only served once they exist on disk
# under backend/models/cvat/paper/{valence,arousal}/{lstm,bilstm}.keras.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "emobank_writer": {
        "processor_cls": EmoBankWriterProcessor,
        "csv": "writer_10240.csv",
        "scale": (1.0, 5.0),
        "language": "en",
        "description": "EmoBank (writer perspective, English) — valence/arousal on SAM 1–5",
    },
    "emobank_reader": {
        "processor_cls": EmoBankReaderProcessor,
        "csv": "reader_10240.csv",
        "scale": (1.0, 5.0),
        "language": "en",
        "description": "EmoBank (reader perspective, English) — valence/arousal on SAM 1–5",
    },
    "fb": {
        "processor_cls": FBProcessor,
        "csv": "fb_dataset_cleaned.csv",
        "scale": (1.0, 9.0),
        "language": "en",
        "description": "Facebook Valence-Arousal Posts — English social media, SAM 1–9",
    },
    "sst": {
        "processor_cls": SSTProcessor,
        "csv": "stanford_sentiment_assembled.csv",
        "scale": (0.0, 1.0),
        "language": "en",
        "description": "Stanford Sentiment Treebank — English movie reviews, valence only (0–1)",
    },
    "cvat": {
        "processor_cls": CVATProcessor,
        "csv": "CVAT.csv",
        "scale": (1.0, 9.0),
        "language": "zh",
        "description": "Chinese Valence-Arousal Text — mixed-domain Chinese, SAM 1–9",
    },
}

ARCHITECTURES: Tuple[str, ...] = ("lstm", "bilstm")
DIMENSIONS: Tuple[str, ...] = ("valence", "arousal")
# Only 'paper' checkpoints are shipped; list kept for forward-compat.
RUN_TAGS: Tuple[str, ...] = ("paper",)
DEFAULT_RUN_TAG: str = "paper"
