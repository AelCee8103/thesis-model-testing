# Enhanced Tree-Structured Regional CNN-LSTM with Bidirectional LSTM for Dimensional Sentiment Analysis

An undergraduate thesis project (BS Computer Science, University of Mindanao,
Davao City) that studies whether replacing the unidirectional LSTM in a
tree-structured regional sentiment model with a **Bidirectional LSTM** improves
continuous valence and arousal prediction.

The repository contains the training pipeline, the trained models, a FastAPI
prediction service, and a React dashboard that lets you score any sentence and
compare the baseline against the proposed model side by side.

> To install and run everything locally, see **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)**.

---

## What this is about

Dimensional sentiment analysis rates text on continuous emotional scales rather
than positive/negative labels:

- **Valence** = how pleasant or unpleasant the text is.
- **Arousal** = how calm or activated it is.

The baseline is the Tree-Structured Regional CNN-LSTM of Wang et al. (2020). It
parses a sentence into a syntax tree, splits the tree into regions, encodes each
region with a small CNN, and then reads the sequence of regions with an LSTM
before predicting a score.

**The hypothesis.** A one-directional LSTM reads regions left to right, so it
cannot use a later clause to resolve an earlier one (for example
*"The setup was hard, but the result is perfect"*). Reading the regions in both
directions with a **BiLSTM** should fix this and produce better scores. This
project changes only that one layer and holds everything else identical, so any
difference is attributable to bidirectionality alone.

## How the model works

```
Sentence
  -> Constituency parse (Stanza)
  -> Split the tree into regions at the dataset's optimal depth
  -> Regional CNN (60 filters, kernel 3) + max-pooling  [per region]
  -> Sequence layer:  LSTM (baseline)  vs  BiLSTM (proposed)
  -> Linear decoder -> valence / arousal score
```

Three architecture variants are implemented so the comparison is fair:

| Variant | Sequence layer | Purpose |
|---|---|---|
| TSR-CNN-LSTM | LSTM, 120 units | Baseline (Wang et al.) |
| TSR-CNN-BiLSTM (full) | BiLSTM, 120 per direction (240) | Proposed |
| TSR-CNN-BiLSTM (matched) | BiLSTM, 60 per direction (120) | Isolates bidirectionality from parameter count |

## Datasets

| Dataset | Language | Scale | Domain |
|---|---|---|---|
| EmoBank writer / reader | English | 1 - 5 | News and fiction sentences |
| FB | English | 1 - 9 | Facebook posts |
| SST | English | 0 - 1 | Movie reviews (valence only) |
| CVAT | Chinese | 1 - 9 | Chinese social forum text |

## Key findings (reported honestly)

Bidirectionality helps in specific conditions, not universally:

- **Chinese valence (CVAT) is the flagship result.** Pearson r improves from
  **0.178 to 0.342** (a 90%+ relative gain), significant at both full and matched
  capacity. Because the effect is stronger at matched capacity, it comes from
  bidirectionality itself, not the extra parameters.
- **English reader valence (EmoBank reader)** shows a small significant gain at
  matched capacity (r 0.226 to 0.252).
- **No benefit** for FB, SST, or arousal at either capacity. An apparent writer
  valence gain at full capacity disappears at matched capacity, so it was driven
  by parameter count, not bidirectionality.

The effect is largest where sentence structure rewards using future context
(Chinese, with topic-prominent word order and sentence-final particles) and
absent where sentiment is resolvable locally at the phrase level.

## Repository layout

```
backend/      FastAPI prediction service + regional_cnn_lstm.py training pipeline
  models/     trained .keras checkpoints served by the API
  cache/      vocab + parsed-tree caches (so the corpora are not re-parsed)
frontend/     React + Vite dashboard (single compare + CSV evaluation)
docs/         USER_MANUAL.md (how to run), methodology.md
THESIS_CONTEXT.md   full project context
```

## The dashboard

The bundled web app has two modes:

- **Single**: enter a sentence, pick a dataset and dimension, and see the
  baseline LSTM and proposed BiLSTM predictions side by side with a score gauge,
  the parsed regions, and reliability warnings.
- **Evaluate CSV**: upload labelled rows and get MAE, Pearson r, and a
  Mann-Whitney U significance test comparing the two models against ground truth.

Run it with the steps in [docs/USER_MANUAL.md](docs/USER_MANUAL.md). In short:
start the backend (`cd backend && make install && make dev`), then the frontend
(`cd frontend && npm install && npm run dev`), and open `http://localhost:5173`.

## Team

- **Ivan M. Eltagonde** (Claudius) — lead developer, primary author
- **Augustine Ervin Leigh C. Cantila** — co-author, AI Engineer
- **Ernest Josh L. Nisperos** — co-author
- **Fe B. Yara, MSIS** — faculty adviser

University of Mindanao, Davao City. BS Computer Science.

## Baseline reference

Wang, J., Yu, L.-C., Lai, K. R., & Zhang, X. (2020). Tree-Structured Regional
CNN-LSTM Model for Dimensional Sentiment Analysis. *IEEE/ACM Transactions on
Audio, Speech, and Language Processing*, 28, 581-591.
doi:10.1109/TASLP.2019.2959251

---

**Disclaimer.** This is a research prototype, not a clinical or diagnostic tool.
Models were trained on specific corpora; predictions on out-of-domain text may be
unreliable.
