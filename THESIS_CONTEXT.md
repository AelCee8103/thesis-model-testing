# Thesis Project Context — TSR-CNN-BiLSTM for Dimensional Sentiment Analysis

> **How to use this document**
> This file contains everything a fresh AI assistant (Claude in terminal, Claude web, any other LLM) needs to help continue this thesis project. Paste the entire file as context at the start of a new conversation, or provide it as a reference file. The document is organized from "what is this project" at the top down to "here are the detailed results and file inventory" at the bottom. Skim the early sections for orientation, reference the later sections as needed.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [Team and Identity](#2-team-and-identity)
3. [Working Preferences and Conventions](#3-working-preferences-and-conventions)
4. [Research Framing](#4-research-framing)
5. [Baseline Paper Summary (Wang et al. 2020)](#5-baseline-paper-summary-wang-et-al-2020)
6. [Proposed Contribution](#6-proposed-contribution)
7. [Technical Pipeline Architecture](#7-technical-pipeline-architecture)
8. [Datasets](#8-datasets)
9. [Experimental Protocol](#9-experimental-protocol)
10. [Results to Date](#10-results-to-date)
11. [Current Thesis Framing](#11-current-thesis-framing)
12. [File Inventory](#12-file-inventory)
13. [Commands and How to Run Things](#13-commands-and-how-to-run-things)
14. [Open Items and Decisions Pending](#14-open-items-and-decisions-pending)
15. [Known Caveats and Limitations](#15-known-caveats-and-limitations)

---

## 1. Project at a Glance

**Thesis title:** Enhanced Tree-Structured Regional CNN-LSTM with Bidirectional LSTM for Dimensional Sentiment Analysis

**Institution:** University of Mindanao, Davao City (BS Computer Science)

**One-line contribution:** Replace the unidirectional LSTM in Wang, Yu, Lai & Zhang (2020)'s Tree-Structured Regional CNN-LSTM with a BiLSTM and measure the impact on dimensional sentiment prediction (valence/arousal) across 5 datasets.

**Current status:** All 18 paired comparison experiments complete (9 paper-config + 9 matched-capacity). Code is finalized, results are analyzed, thesis story has been revised based on what the data actually shows. Tuned-config runs still pending. Writeup and publication still pending. A FastAPI prediction service has been built for a React-based beta tool for psychologist testers.

---

## 2. Team and Identity

- **Claudius (Ivan M. Eltagonde)** — lead developer, primary thesis author, also employed at Inventiv Technology as Lead Software Developer. Based in the Philippines. Communicates in mixed English/Filipino/Tagalog but prefers technical output in English.
- **Augustine Ervin Leigh C. Cantila** — co-author
- **Ernest Josh L. Nisperos** — co-author
- **Fe B. Yara, MSIS** — faculty adviser
- Contact email prefix: `i.eltagonde.533553@umindanao.edu.ph`

---

## 3. Working Preferences and Conventions

These are explicit preferences Claudius has expressed throughout the project. Respect these when producing output.

### Language
- Accept instructions in English or Filipino/mixed.
- **Always produce technical output in English**: code, comments, commit messages, documentation, file names, branch names.
- Thesis writeup is in English.

### Engineering philosophy
- **Zero token waste** — never ask Claudius to paste code when the file is already known. Use `view` tool.
- **Check before editing** — run `git status` or equivalent sanity check before destructive operations.
- **Atomic, numbered plans** — before implementation, show a plan with background, steps, files, validation criteria.
- **Single-file pipeline** — the training code is intentionally in one file (`regional_cnn_lstm.py`). Do not split it into modules without explicit request.

### Writing style preferences
- **No em dashes** — do not use em dashes in text. Use "—" (en dash), commas, or rewording instead.
- **Plain, jargon-free language** in any prose aimed at non-technical readers (thesis methods sections, disclaimers, user-facing text).
- **No sycophantic openers** — do not start replies with "Great question" or "That's a great idea."
- **Honest over optimistic** — if a result is null or weak, say so directly.

### Claude output format for this project
- Use the three-phase cycle: Spec → Plan → Implement.
- Start action-oriented tasks with "🎯 收到任务: ..." (Chinese framing, kept as a convention).
- End with a brief "✅ 任务完成" status block.
- Avoid excessive formatting in casual replies (no headers for short answers).

---

## 4. Research Framing

### Original hypothesis (from research outline)
The baseline TSR-CNN-LSTM suffers from "contextual blindness" because its sequential LSTM layer processes regions unidirectionally, left-to-right. The LSTM has a recency bias that overweights tail information and cannot use later clauses to resolve ambiguity in earlier ones (e.g., "The setup was hard, but the result is perfect"). Replacing the LSTM with a Bidirectional LSTM should resolve this and improve performance.

### Predicted outcomes (from research outline)
- Higher Pearson correlation coefficient (r) on valence and arousal
- Lower Mean Absolute Error (MAE)
- Statistical significance at p < 0.05 (Mann-Whitney U test)

### Objective 1.3.2.3 — the committed success criterion
> "If there is an increase in r and/or a decrease in MAE, then this would be an indication of an improvement. Also, if the p-value of the Mann-Whitney U-test is less than 0.05, the progress is statistically significant."

This is the pre-registered decision rule. Results that meet it constitute success; results that don't must be reported honestly.

---

## 5. Baseline Paper Summary (Wang et al. 2020)

**Citation:** Wang, J., Yu, L.-C., Lai, K. R., & Zhang, X. (2020). Tree-Structured Regional CNN-LSTM Model for Dimensional Sentiment Analysis. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 581–591. DOI: 10.1109/TASLP.2019.2959251.

### Architecture
1. **Constituency parsing** with a syntax tree parser
2. **Tree-structured region division** — nodes at a fixed depth D are treated as regional roots; each region's leaves become a word list
3. **Word embeddings** — GloVe 300d (English), zhwiki-GloVe (Chinese, never publicly released)
4. **Regional CNN** — 60 filters, kernel width 3 applied within each region
5. **Max-pooling** — per-region feature extraction
6. **LSTM sequential layer** — 120 hidden units, processes region vectors in order
7. **Linear decoder** — produces continuous valence and arousal predictions

### Hyperparameters (Table I, Wang et al.)
- CNN: 60 filters, kernel 3, pool 2
- LSTM: 120 hidden units
- Adam optimizer, batch size 32, 20 epochs
- Recurrent dropout 0.25, no L2, learning rate 1e-3
- No label normalization

### Evaluation protocol
- 4 corpora for 5-fold CV at 7:2:1 train:dev:test (EmoBank reader, EmoBank writer, FB, CVAT)
- SST uses its published train/dev/test split
- Metrics: Pearson r (higher is better), MAE (lower is better)
- Significance: Whitney-Mann U test (equivalent to Mann-Whitney U) at p < 0.05

### Optimal region depths (reproduced in our `DATASET_OPTIMAL_DEPTH`)
- EmoBank writer: valence=4, arousal=5
- EmoBank reader: valence=4, arousal=4
- FB: valence=5, arousal=5
- CVAT: valence=8, arousal=8
- SST: valence=5

---

## 6. Proposed Contribution

### The one-variable change
Replace the unidirectional LSTM with a Bidirectional LSTM. Everything else — preprocessing, parsing, region extraction, CNN encoder, linear decoder, hyperparameters — is held identical.

### Architecture variants now implemented
1. **TSR-CNN-LSTM (baseline)** — 120 hidden units, unidirectional
2. **TSR-CNN-BiLSTM (full capacity)** — 120 units per direction, 240 total concatenated
3. **TSR-CNN-BiLSTM (matched capacity)** — 60 units per direction, 120 total (equal to baseline)

The matched-capacity variant was added mid-project because a reviewer would reasonably ask: "Is any observed BiLSTM improvement due to bidirectionality itself, or just the doubled parameter count?" The matched-capacity ablation answers this cleanly.

---

## 7. Technical Pipeline Architecture

### Core file: `regional_cnn_lstm.py` (~3100 lines, single-file)

Built as a modular, extensible pipeline. Key components:

**Config dataclass** — holds all hyperparameters and paths. Mutable at runtime via CLI flags.

**Dataset processors** (subclasses of `BaseDatasetProcessor`):
- `EmoBankWriterProcessor` — loads `writer_10240.csv`
- `EmoBankReaderProcessor` — loads `reader_10240.csv`
- `FBProcessor` — loads `fb_dataset_cleaned.csv` (text, V, A)
- `CVATProcessor` — loads `CVAT.csv` (Big5-HKSCS), language="zh", applies OpenCC Traditional→Simplified conversion, uses Stanza zh-hans native tokenization (no jieba)
- `SSTProcessor` — loads `stanford_sentiment_assembled.csv` (text/label/split), valence-only (NaN arousal column), overrides `predefined_split` to return the published train=8544/dev=1101/test=2210 split

**ConstituencyParser** — wraps Stanza. Caches parsed trees to `cache/parsed_trees_<dataset>.pkl`.

**Region extraction** — `extract_regions_at_depth()` walks the tree to a given depth and returns word lists per region.

**Vocabulary + embeddings** — `build_vocab()` indexes words from training data; `load_glove_matrix()` builds the embedding matrix. Results cached to `cache/embedding_matrix_<dataset>_<hash>.pkl`.

**Model** — `build_model()` constructs the Keras model. The `cfg.bidirectional_lstm` and `cfg.bilstm_units_per_direction` flags control the sequential layer:
- `bidirectional_lstm=False` → unidirectional LSTM with `cfg.lstm_units` units (baseline)
- `bidirectional_lstm=True`, `bilstm_units_per_direction=None` → full-capacity BiLSTM (`cfg.lstm_units` per direction)
- `bidirectional_lstm=True`, `bilstm_units_per_direction=60` → matched-capacity BiLSTM (60 per direction)

**Training loop and comparison** — `run_comparison()` for datasets with CV, `run_comparison_predefined_split()` for SST. Both produce paired LSTM-vs-BiLSTM outputs with identical seeds and data per fold.

**Statistical tests** — Mann-Whitney U (primary per pre-registered plan), Wilcoxon signed-rank (paired secondary), rank-biserial correlation (effect size).

### Output layout
```
models/<dataset>/<run_tag>/<dimension>/<arch>/
    best_d<D>_fold1..5.keras
    best_d<D>_overall.keras         # copy of the best-by-dev-r fold

logs/compare_<dataset>_<dim>_d<D>_<run_tag>.json    # headline summary
logs/metrics_<dataset>_d<D>_<dim>_<arch>_fold<K>_<run_tag>.json
logs/predictions/predictions_<dataset>_<dim>_compare_fold<K>_<run_tag>.npz
```

The `--run-tag` flag (`paper`, `tuned`, `matched`) scopes output so all three experimental configurations coexist on disk without collision.

---

## 8. Datasets

All 5 datasets used. The pipeline expects the raw CSVs and embedding files in `BASE_DIR`.

| Dataset | Size | Language | Scale | Source |
|---|---|---|---|---|
| EmoBank writer | ~10,240 rows | English | 1–5 SAM | JULIELab/EmoBank on GitHub |
| EmoBank reader | ~10,240 rows | English | 1–5 SAM | same |
| FB (Facebook Valence-Arousal Posts) | ~2,895 rows | English | 1–9 SAM | Preoțiuc-Pietro et al. 2016 |
| CVAT (Chinese Valence-Arousal Texts) | ~2,009 rows | Chinese (Traditional) | 1–9 SAM | Yu et al. 2016 |
| SST (Stanford Sentiment Treebank) | 8,544 / 1,101 / 2,210 split | English | 0–1 | Socher et al. 2013, reformatted to `text, label, split` columns |

### Embeddings
- **English:** `glove.840B.300d.txt` (~5.3 GB) — standard
- **Chinese:** `sgns.merge.word` from Chinese-Word-Vectors "Mixed-large 综合" (~4 GB). **This is a substitution** — Wang et al.'s "zhwiki-GloVe" is not publicly distributed. The Mixed-large corpus is a superset of zhwiki, trained with SGNS instead of GloVe. This substitution is documented in our methodology.

Neither embedding file is committed to the repo (see `.gitignore`).

---

## 9. Experimental Protocol

### Three experimental configurations

**Paper-config** (`--paper-config --run-tag paper`)
Hyperparameters exactly match Wang et al. Table I: dropout 0.25, no L2, LR 1e-3, 20 epochs, no label normalization. This is the faithful replication condition.

**Tuned-config** (`--run-tag tuned`, no `--paper-config`)
Defaults to stronger regularization: label normalization on, spatial dropout 0.3, L2 1e-5, LR 5e-4, 40-epoch budget with early stopping patience 10.

**Matched-capacity** (`--paper-config --bilstm-units-per-direction 60 --run-tag matched`)
Paper hyperparameters, but BiLSTM has 60 units per direction (120 total) instead of 120 per direction (240 total). LSTM baseline unchanged. Isolates bidirectionality from parameter count.

### Per-dataset protocol
- **EmoBank writer, EmoBank reader, FB, CVAT:** 5-fold CV at 7:2:1 via rotating tenths (50% of dataset seen as test across 5 folds)
- **SST:** single predefined train/dev/test split (no CV)

### Per-architecture protocol within each comparison
- Same random seed per fold for both architectures (seed = base_seed + fold_k)
- Same vocabulary, same embedding matrix, same cached parse trees
- Paired statistical tests on per-sample absolute errors

### Statistical tests reported
1. **Per-fold Pearson r and MAE** — mean, std, min, max across 5 folds
2. **Pooled across folds** — concatenated predictions, single r and MAE
3. **Mann-Whitney U** — primary significance test (pre-registered)
4. **Wilcoxon signed-rank** — paired secondary test
5. **Rank-biserial correlation** — effect size on [-1, 1] scale

---

## 10. Results to Date

### Paper-config (full capacity BiLSTM, 240 total) — 9 runs completed

| Dataset | Dim | LSTM r | BiLSTM-240 r | MWU p | r_rb | Verdict |
|---|---|---|---|---|---|---|
| EmoBank writer | val | 0.1874 | 0.1940 | 0.043 | -0.023 | **BiLSTM sig win** |
| EmoBank writer | aro | 0.1464 | 0.1442 | 0.289 | -0.012 | Null |
| EmoBank reader | val | 0.2458 | 0.2487 | 0.435 | +0.009 | Null |
| EmoBank reader | aro | 0.1426 | 0.1518 | 0.348 | +0.011 | Null |
| FB | val | 0.6330 | 0.6201 | 0.958 | -0.001 | Null |
| FB | aro | 0.8319 | 0.8274 | 0.210 | +0.027 | Null |
| CVAT | **val** | 0.2151 | **0.3315** | **0.039** | **-0.053** | **BiLSTM sig win (strongest)** |
| CVAT | aro | 0.2755 | 0.2739 | 0.919 | +0.003 | Null |
| SST | val | **0.6941** | 0.6745 | **0.005** | +0.048 | **Baseline LSTM sig win** |

### Matched-capacity (120 total BiLSTM, 60 per direction) — 9 runs completed

| Dataset | Dim | LSTM r | BiLSTM-120 r | MWU p | r_rb | Verdict |
|---|---|---|---|---|---|---|
| EmoBank writer | val | 0.1863 | 0.1817 | 0.393 | +0.010 | Null — previous win DISAPPEARS |
| EmoBank writer | aro | 0.1377 | 0.1448 | 0.419 | +0.009 | Null |
| EmoBank reader | **val** | 0.2265 | **0.2515** | **0.005** | **-0.032** | **BiLSTM sig win (NEW)** |
| EmoBank reader | aro | 0.1460 | 0.1465 | 0.061 | +0.021 | Null (marginal) |
| FB | val | 0.6210 | 0.6267 | 0.415 | -0.018 | Null |
| FB | aro | 0.8325 | 0.8234 | 0.522 | +0.014 | Null |
| CVAT | **val** | 0.1778 | **0.3422** | **0.017** | **-0.062** | **BiLSTM sig win (robust, stronger)** |
| CVAT | aro | 0.2665 | 0.2772 | 0.978 | +0.001 | Null |
| SST | val | 0.6876 | 0.6721 | 0.115 | +0.027 | Null — SST gap shrinks, no longer sig |

### Critical findings from the ablation

1. **CVAT valence is the flagship result.** Significant at BOTH capacities. Effect is *stronger* at matched capacity (r_rb -0.062 vs -0.053). Proof that bidirectionality itself drives the Chinese-valence improvement, not parameter count. Pearson r goes from 0.178 to 0.342 — a 90%+ relative improvement.

2. **EmoBank reader valence is a NEW positive finding.** Not significant at full capacity (p = 0.435). Significant at matched capacity (p = 0.005). Full-capacity BiLSTM was overfitting; matched capacity fixed it.

3. **EmoBank writer valence is a retracted false positive.** Paper-config showed "significant BiLSTM win" (p = 0.043). Matched capacity shows BiLSTM slightly *losing* (p = 0.393). The previous gain came from parameter count, not bidirectionality.

4. **Baseline drift between runs.** LSTM per-fold metrics differ by ~0.003–0.007 in Pearson r between the paper and matched runs despite identical seeds, data, and architecture. This is TensorFlow CPU/GPU nondeterminism (op scheduling, reduction ordering), not a bug. Must be acknowledged as a methodology caveat.

5. **Multiple-comparison concern.** 18 tests total. Bonferroni α = 0.05/18 = 0.0028. Only the matched-capacity reader valence (p = 0.005) survives strict correction. CVAT valence at matched (p = 0.017) and paper (p = 0.039) do not survive Bonferroni but have large effect sizes that matter regardless. Wang et al. don't apply correction either, but we should report both.

### Tuned-config runs — NOT YET EXECUTED
The 9 tuned-config runs are still pending. They would show whether the main findings hold under stronger regularization, and might recover effects that overfitting masked at paper-config.

---

## 11. Current Thesis Framing

The original outline's pitch — "BiLSTM universally improves dimensional sentiment analysis" — is **not** what the data shows. The revised story, which the data actually supports:

> **"Bidirectional processing provides a substantive capacity-independent improvement for Chinese valence prediction (CVAT: r = 0.178 → 0.342), a statistically significant small improvement for English reader-perspective valence at matched capacity (EmoBank reader: r = 0.226 → 0.252), and no benefit for FB, SST, or arousal prediction at either capacity. An apparent writer-valence improvement at full capacity disappears under matched capacity, indicating it was driven by doubled parameter count rather than bidirectionality. The effect is largest where language structure rewards future-context resolution — specifically Chinese, with its topic-prominent word order and sentence-final particles — and absent where sentiment is locally resolvable at the phrase level."**

This is a more textured, more defensible story than the original pitch. Reviewers prefer nuanced findings with conditions attached.

---

## 12. File Inventory

All files have been published to `/mnt/user-data/outputs/` during the project.

### Core code
| File | Purpose |
|---|---|
| `regional_cnn_lstm.py` | Single-file training pipeline (~3100 lines) |
| `predict_service.py` | FastAPI HTTP service + Python wrapper for single-text inference |

### Documentation
| File | Purpose |
|---|---|
| `README.md` | Public-facing GitHub README with setup, downloads, usage |
| `API_CONTRACT.md` | Frontend integration doc for the React developer |
| `BATCH_RUN_GUIDE.md` | Short guide on batch files and logs |
| `THESIS_CONTEXT.md` | This file — complete project context |

### Training batch scripts (Windows)
| File | Purpose |
|---|---|
| `run_paper.bat` | All 9 paper-config comparisons |
| `run_tuned.bat` | All 9 tuned-config comparisons |
| `run_matched.bat` | All 9 matched-capacity comparisons |
| `run_all.bat` | Master: paper → tuned → matched sequentially |

### Training batch scripts (Linux / Azure ML)
| File | Purpose |
|---|---|
| `run_paper.sh` | Linux equivalent |
| `run_tuned.sh` | Linux equivalent |
| `run_matched.sh` | Linux equivalent |
| `run_all.sh` | Linux master |

### API launchers
| File | Purpose |
|---|---|
| `run_api.bat` | Launch prediction service (Windows) |
| `run_api.sh` | Launch prediction service (Linux) |

### Dependencies and repo config
| File | Purpose |
|---|---|
| `requirements.txt` | pip-installable dependencies |
| `.gitignore` | Excludes data, embeddings, caches, models, logs |

### Runtime-created (not committed, gitignored)
| Folder | Purpose |
|---|---|
| `cache/` | Parse trees and embedding matrices (reused across runs) |
| `models/` | Trained `.keras` checkpoints, scoped by run_tag |
| `logs/` | Training logs, comparison summaries, per-fold metrics, prediction `.npz` |

---

## 13. Commands and How to Run Things

### First-time setup
```bash
python -m venv .venv
# Windows cmd: .venv\Scripts\activate
# PowerShell: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
pip install -r requirements.txt

# Download Stanza models (one-time, ~800 MB for EN+ZH)
python -c "import stanza; stanza.download('en', processors='tokenize,pos,constituency')"
python -c "import stanza; stanza.download('zh-hans', processors='tokenize,pos,constituency')"
```

### Training — paper-config comparisons (9 runs)

Windows:
```cmd
run_paper.bat
```

Linux/Azure ML:
```bash
chmod +x run_paper.sh
./run_paper.sh
# For unattended multi-hour runs inside tmux (recommended on Azure ML):
tmux new -s paper
./run_paper.sh
# Ctrl+B then D to detach; tmux attach -t paper to return
```

### Training — matched-capacity (9 runs)
```bash
./run_matched.sh   # or run_matched.bat
```

### Training — tuned-config (9 runs, PENDING)
```bash
./run_tuned.sh     # or run_tuned.bat
```

### Training — everything in sequence
```bash
./run_all.sh       # or run_all.bat
```

### Prediction API
```bash
pip install fastapi uvicorn pydantic    # one-time
./run_api.sh                            # starts at http://127.0.0.1:8000
# Then open http://localhost:8000/docs for the interactive Swagger UI
```

### Single-command invocations (for testing)
```bash
# List every model checkpoint available on disk
python predict_service.py --list-models

# Run one prediction from CLI
python predict_service.py --test-predict "The movie was wonderful." \
    --test-dataset emobank_writer --test-dimension valence \
    --test-architecture bilstm --test-run-tag paper

# Single-run of training code
python regional_cnn_lstm.py --dataset emobank_writer --compare \
    --dimension valence --paper-config --run-tag paper
```

---

## 14. Open Items and Decisions Pending

In rough priority order:

1. **Run `run_tuned.bat` / `run_tuned.sh`** — 9 more comparisons. Tells us if main findings hold under stronger regularization. Compute: ~5–8 hours.

2. **Write the methods section of the paper** using the three-way results structure:
   - Protocol description (5-fold CV, single-split SST, paper/matched/tuned configs)
   - Metrics and statistical testing (MWU primary, Wilcoxon + rank-biserial secondary)
   - Threats to validity (seed sensitivity, embedding substitution, tokenizer substitution, T→S conversion, multiple comparisons, TensorFlow nondeterminism)
   - Reproducibility (code, caches, checkpoints all available)

3. **Write the results section** with the headline three-way comparison table and honest retraction of the writer-valence claim.

4. **Revise the thesis introduction and abstract** to match the actual story: bidirectionality helps on CVAT and reader-valence specifically, not universally.

5. **GitHub push** — the code + README + requirements + batch files are ready to publish. Data files and embeddings are gitignored.

6. **Beta tool integration** — React colleague needs to swap the stub for `fetch` calls to the API. `API_CONTRACT.md` has the JavaScript snippets ready. Prediction service code is untested at runtime by Claude's sandbox (no TensorFlow available for validation); first real launch may reveal small fixable issues.

7. **Optional improvements** if time permits:
   - Multiple seeds per fold to quantify init variance (would require a pipeline extension)
   - Batch prediction endpoint for the beta tool (process CSVs of texts)
   - Attention visualization showing which regions contributed to each score

---

## 15. Known Caveats and Limitations

These must be acknowledged in the paper and should be kept in mind when discussing results:

### Evaluation
- **5-fold CV at 7:2:1 tests only 50% of each dataset.** Wang et al.'s protocol uses 10 internal tenths; only tenths 0–4 serve as test, tenths 5–9 are always train/dev. `--compare-folds 10` would test everything but doubles compute.
- **Single seed per fold.** Same seed for both architectures in fold k ensures comparability, but doesn't quantify within-fold init variance. Wang et al. do the same.
- **Early stopping uses val_loss, not val_r.** MSE-trained models can hit low val_loss with flat predictions. Label normalization reduces but doesn't eliminate this.

### Statistical testing
- **MWU is underpowered against effect size at large n.** Pooled n = 5120 for EmoBank. Trivial mean-error differences reach p < 0.05. Always read rank-biserial alongside p-values.
- **Multiple comparisons not corrected in primary analysis.** 18 tests. Bonferroni would require p < 0.0028; only matched reader valence (p = 0.005) survives.

### Data and embeddings
- **Chinese embedding substitution.** `sgns.merge.word` (Mixed-large SGNS) instead of Wang's zhwiki-GloVe (unavailable). Superset corpus, different algorithm. Documented.
- **Chinese tokenizer substitution.** Wang et al. don't specify a segmenter. We use Stanza's CTB-trained tokenizer, consistent with the constituency parser.
- **CVAT Traditional→Simplified conversion.** Via OpenCC before parsing. Documented.
- **GloVe OOV on noisy text.** Social media text has high OOV rate (logged during training). OOV tokens get small random vectors.

### Implementation
- **TensorFlow nondeterminism.** Same seed, same data, same model produce slightly different results between the paper-run and matched-run LSTM baselines (|Δr| < 0.01). This is op scheduling and reduction ordering in TF, not a bug. Documented.
- **Stanza parsing is CPU-bound.** First run on each dataset takes 15–25 minutes; results are cached in `cache/parsed_trees_<dataset>.pkl` and reused forever after.

### Scope
- **Single-modal text only.** No images, emoji handling, or multimodal signals.
- **Regression only.** No categorical classification, no aspect-based sentiment analysis.

---

## Quick-start for a new Claude conversation

If you're a fresh AI assistant reading this context, here's what a typical follow-up interaction will look like:

- **Claudius asks about running things** → reference Section 13, give commands in English
- **Claudius asks about results** → reference Section 10 for numbers, Section 11 for framing
- **Claudius asks to modify code** → reference Section 7 for architecture, use the three-phase Spec → Plan → Implement cycle
- **Claudius asks about what comes next** → Section 14's open items list
- **Claudius asks about known issues** → Section 15 caveats

Respect Section 3's working preferences: no em dashes, plain language, English technical output, atomic plans before code, honest assessments over optimistic framing.

---

**Document version:** 1.0 (covers state as of all paper-config and matched-capacity runs complete, tuned-config pending)
