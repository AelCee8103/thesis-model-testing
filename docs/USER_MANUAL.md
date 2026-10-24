# User Manual: Running the Project Locally

Get the TSR-CNN-BiLSTM sentiment tool running on one machine. Two processes talk
over HTTP:

```
Browser -> Frontend (Vite, :5173) -> Backend (FastAPI, :8000) -> Stanza + Keras models
```

Start the **backend first**, then the frontend. Everything the models need is
already in the repo, so no training, cache building, or embedding downloads are
required for a normal run.

## Prerequisites

- Python 3.11 or 3.12
- Node.js 20+ and npm
- git, and ~5 GB free disk (TensorFlow + Stanza models)

CPU is fine; a GPU is optional.

## Backend

From `backend/`:

```bash
cd backend

# 1. Install (uses python3.11; override if needed)
make install                    # or: make install PYTHON=python3.12

# 2. Configure environment
cp .env.example .env            # then edit .env (see below)

# 3. Download the English parser (Chinese only needed for CVAT)
make stanza-en                  # make stanza-zh  for CVAT

# 4. Run
make dev                        # uvicorn app.main:app on 127.0.0.1:8000
```

In `.env`, set `BASE_DIR` to the absolute path of `backend/` (the example ships a
Windows path) and choose the device:

```
BASE_DIR=/absolute/path/to/Model/backend
PORT=8000
CUDA_VISIBLE_DEVICES=-1          # -1 = CPU, 0 = first GPU
```

No-Makefile equivalent for install and run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Verify** (in a second terminal, from `backend/`):

```bash
make health          # {"status": "ok"}
make models          # JSON list of checkpoints
make test-predict    # a prediction with a score
```

Interactive API docs: `http://127.0.0.1:8000/docs`. The first prediction to a
model takes ~10 seconds while it loads; later ones are ~200-500 ms.

## Frontend

Keep the backend running. From `frontend/`:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The backend URL comes from `frontend/.env` (`VITE_API_URL=http://127.0.0.1:8000`,
already set). Change it and restart `npm run dev` to point at a different
backend. When the page loads the dataset list, it has reached the backend.

## Using the app

Open `http://localhost:5173`. The top toggle switches between **Single** and
**Evaluate CSV**. Every prediction runs both architectures automatically: the
baseline **LSTM** and the proposed **BiLSTM**.

Datasets you can pick, with their rating scales:

| Dataset | Language | Scale | Dimensions |
|---|---|---|---|
| emobank_writer / emobank_reader | English | 1.0 - 5.0 | valence, arousal |
| fb | English | 1.0 - 9.0 | valence, arousal |
| sst | English | 0.0 - 1.0 | valence only |
| cvat | Chinese | 1.0 - 9.0 | valence, arousal |

### Single mode: score one sentence

1. In the **Input** card, type or paste text (up to 10,000 characters). Use
   **Try a sample** to drop in an example for the chosen dataset.
2. Under **Setup**, pick a **Dataset** and a **Dimension** (Valence =
   pleasantness, Arousal = activation). The line below shows that dataset's scale
   and region depth.
3. Click **Compare baseline vs proposed**.

The **Comparison** card then shows, for the same input:

- A verdict badge: *BiLSTM scores higher*, *LSTM scores higher*, or *Models
  agree*.
- Two score cards, **Baseline (LSTM)** and **New Model (BiLSTM)**, each with the
  score on the dataset scale, a plain-language label, a gauge, and the `raw`
  model output. The BiLSTM card also shows the delta versus the baseline.
- The plain-language label comes from where the score sits on the scale. For
  valence: Strongly negative -> Mildly negative -> Neutral -> Mildly positive ->
  Strongly positive. For arousal: Very calm -> Calm -> Moderate -> Activated ->
  Highly aroused.
- Meta tiles (region depth, regions used vs capacity, token count), the parsed
  **regions** the model saw, and any **warnings** (for example a high
  out-of-vocabulary rate, or text truncated because it produced too many
  regions).

### Evaluate CSV mode: score many rows and compare accuracy

1. Click **Choose CSV** and select a file. It needs a `text` column plus one of
   `true_valence` or `true_arousal`. The truth column sets the dimension
   automatically. Limits: 2,000 rows, 10,000 characters per row. Encoding
   (UTF-8, Big5, GB18030, UTF-16) is detected for you.

   ```csv
   text,true_valence
   I absolutely loved this film.,4.6
   It was a dull and forgettable evening.,2.1
   ```

2. Under **Setup**, pick the **Dataset** whose scale matches your truth values.
   The app warns if the dataset has no model for the file's dimension, or if
   truth values fall outside the dataset scale.
3. Click **Run N rows**. A progress bar tracks scoring; **Cancel** stops it.

When scoring finishes, the **Evaluation** card reports, over all successfully
scored rows (`n`):

- **MAE** (lower is better) and **Pearson r** (higher is better) for each model,
  with the winning value highlighted.
- A **Mann-Whitney U** test on the two models' absolute errors, with the
  `p-value` and a *Significant* / *Not significant* badge (significant means
  `p < 0.05`). When significant, it names which model performed better.

The **Results** table below lists every row with both scores, the delta, and the
absolute error against truth, so you can inspect individual disagreements.

> Research prototype, not for clinical or diagnostic use. Predictions on text
> unlike a model's training domain may be unreliable.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3.11: command not found` | `make install PYTHON=python3.12`, or use the plain-venv steps above. |
| Port 8000 in use | `make dev PORT=8001`, then update `frontend/.env`. |
| Frontend: "Could not load models" | Backend not running or wrong `VITE_API_URL`. Confirm `make health` is ok; restart `npm run dev` after editing `.env`. |
| `/predict` 404 about a missing vocab cache | A `cache/vocab_<dataset>.pkl` was removed. Rebuild: `.venv/bin/python generate_cache.py`. |
| First prediction hangs ~10s | Expected; the model is loading. Later calls are fast. |
