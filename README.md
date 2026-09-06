<div align="center">

# Fraud Detection, SQL + Supervised ML
<img width="1733" height="908" alt="Fraud-Detection-SQL-Supervised" src="https://github.com/user-attachments/assets/5a18aecb-2305-42dd-a754-70a850156831" />

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![SQLite](https://img.shields.io/badge/SQLite-Feature%20Engineering-003B57)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LogReg%20%2B%20RandomForest-orange)
![Responsible ML](https://img.shields.io/badge/Responsible%20ML-Leakage%20Analysis-green)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Supervised/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Supervised/actions/workflows/ci.yml)

</div>

A responsible machine learning project that turns labeled transactions into a **fraud-risk ranking**, using **SQL (SQLite) window-function feature engineering** and a **Logistic Regression / Random Forest** pipeline with **point-in-time (leakage-safe) features**, **honest evaluation**, **configurable decision thresholds**, and **command-line training and inference**.

> **Important:** This project is a **fraud-risk ranking pipeline and educational demo**, not a production fraud system.
>
> The model, thresholds, and reports are designed to demonstrate a professional, leakage-aware, SQL-plus-Python classification workflow. They rank transactions by estimated fraud probability using historical behavior features; they do not verify transactions against external evidence, do not block or approve payments, and should not be used for automated fraud decisions, account actions, or any high-stakes financial decision.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Leakage Controls and Honest Evaluation](#leakage-controls-and-honest-evaluation)
- [Model Output](#model-output)
- [Model Artifacts and Loading Safety](#model-artifacts-and-loading-safety)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Fraud detection is often presented as if a classifier can decide whether a transaction is fraudulent. In reality, a model trained on historical labels cannot verify intent, and its apparent accuracy can come from data leakage rather than genuine signal. A fraud score is only useful if it can support a defensible action:

- rank transactions by estimated fraud risk for human review
- compute features only from information that would actually be available at scoring time
- expose how a naive feature design would have leaked future information into the past
- communicate precision/recall trade-offs honestly under severe class imbalance

This project demonstrates an end-to-end, honest SQL-plus-Python classification workflow on a labeled transactions dataset. It includes SQL window-function feature engineering, point-in-time leakage controls, model training and evaluation, configurable decision thresholds, an opt-in alternative classifier for comparison, and a fully automated quality/test gate.

The goal is to show how a fraud model can be turned into a **responsible ranking tool for review**, not just a single accuracy or AUC number.

---

## What This Project Does

This project can:

- Load a labeled transactions CSV into a SQLite database
- Engineer per-user and per-day behavioral features with SQL window functions
- Compute every feature **point-in-time**, using only a user's *prior* transactions
- Train a Logistic Regression model (default) or a Random Forest (opt-in) on those features
- Score every transaction with a fraud probability and rank them by risk
- Report AUC, precision, recall, and F1 at a configurable decision threshold
- Generate an ROC curve chart
- Validate CLI inputs (threshold, test size) and fail with a clear error instead of a confusing one
- Persist the fitted scaler + classifier via `joblib` for reuse without retraining
- Run automated tests and a GitHub Actions CI quality gate on every push

---

## What This Project Does Not Do

This project does **not**:

- Prove that any specific transaction is fraudulent
- Score transactions online or integrate with a live payments system
- Replace a fraud-operations team or case-management workflow
- Detect all types of fraud (this dataset and feature set target one behavioral pattern)
- Guarantee real-world precision on data unlike the training distribution
- Approve, decline, or block transactions on its own

A real production fraud system would need online scoring, a model registry, drift monitoring, human-in-the-loop review, and much richer features than this project's scope.

---

## Key Features

- **SQL (SQLite) window-function feature engineering** for per-user and per-day transaction statistics
- **Point-in-time features**: every aggregate uses `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`, so a transaction never sees its own amount or anything in the future
- **Logistic Regression** baseline with **balanced class weights** for rare fraud cases
- **Opt-in Random Forest** (`--model random_forest`) for comparison against the default
- **Configurable, validated CLI**: threshold, test size, random state, output directory, and model are all flags or environment variables, with range validation on threshold/test-size
- **CSV schema validation** at load time, so a malformed CSV fails with a clear message instead of a confusing downstream error
- **Ranked fraud-risk outputs**: a per-transaction score and a per-user summary
- **Optional model persistence** via `joblib`, decoupled from training
- **Unit tests and GitHub Actions CI**, with a 90%-minimum coverage gate
- **Pinned runtime and dev dependencies**, `pre-commit`, and `Dependabot`

---

## System Workflow

```text
Raw transactions CSV
        ↓
Load into SQLite (create_db.py, with column validation)
        ↓
SQL feature engineering (queries.sql, point-in-time window functions)
        ↓
Train/test split + StandardScaler
        ↓
Classifier (Logistic Regression default, Random Forest optional)
        ↓
Probability + threshold-based precision/recall/F1
        ↓
ROC curve, metrics.json
        ↓
Ranked fraud_scores.csv and per-user fraud_summary.csv
```

---

## Project Structure

```text
Fraud-Detection-SQL-Supervised/
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   └── dependabot.yml
│
├── data/
│   └── transactions_labeled.csv
│
├── outputs/
│   ├── charts/
│   │   └── roc_curve.png
│   ├── metrics.json
│   ├── fraud_scores.csv
│   └── fraud_summary.csv
│
├── src/
│   ├── create_db.py
│   ├── queries.sql
│   ├── train_supervised.py
│   └── utils.py
│
├── tests/
│   ├── conftest.py
│   ├── test_create_db.py
│   ├── test_train_supervised.py
│   └── test_utils.py
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Outputs are generated, not committed (see `.gitignore`); regenerate them locally or download them from the `pipeline-outputs` artifact on any CI run.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Supervised.git
cd Fraud-Detection-SQL-Supervised
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (pytest, Ruff, Black, mypy):

```bash
pip install -r requirements-dev.txt
```

The project is also pip-installable (`pip install -e .`), which exposes `fraud-create-db` and `fraud-train` console scripts equivalent to running the scripts under `src/` directly.

---

## Quick Start

Load labeled data into SQLite:

```bash
python src/create_db.py --csv data/transactions_labeled.csv --db fraud.db
```

Train and evaluate the model:

```bash
python src/train_supervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

Train the opt-in Random Forest instead of the default Logistic Regression:

```bash
python src/train_supervised.py --db fraud.db --sql src/queries.sql --outdir outputs --model random_forest
```

---

## Training and Evaluation

The training script runs the SQL feature query, splits the result into train/test, scales features, trains the classifier, evaluates on the holdout set, ranks every transaction by fraud probability, and writes metrics, ranked outputs, and a chart.

```bash
python src/train_supervised.py \
  --db fraud.db \
  --sql src/queries.sql \
  --outdir outputs \
  --threshold 0.5 \
  --test-size 0.25 \
  --random-state 42
```

Every path and parameter can also be set via environment variable instead of a flag: `FRAUD_CSV_PATH`, `FRAUD_DB_PATH`, `FRAUD_SQL_PATH`, `FRAUD_OUTDIR`, `FRAUD_THRESHOLD`, `FRAUD_TEST_SIZE`, `FRAUD_RANDOM_STATE`, `FRAUD_MODEL_PATH`, `FRAUD_MODEL`. A flag always overrides the matching env var. `--threshold` must be in `[0, 1]` and `--test-size` in `(0, 1)`; an invalid value from either the flag or the env var fails fast with a clear error instead of a confusing one deep inside scikit-learn.

Generated outputs include:

```text
outputs/metrics.json
outputs/fraud_scores.csv
outputs/fraud_summary.csv
outputs/charts/roc_curve.png
```

---

## Leakage Controls and Honest Evaluation

The feature query originally aggregated each user's transaction statistics over the **entire dataset**, including transactions that happen *after* the one being scored. That is look-ahead leakage: a real-time fraud system would never have access to a user's future transactions when scoring the current one.

`src/queries.sql` now computes every aggregate **point-in-time**, using SQLite window functions:

```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
```

partitioned by user (and by user + day for the daily features), ordered by date and transaction ID. A regression test (`test_queries_sql_features_are_point_in_time`) checks this directly: every user's first transaction must show `tx_count == 0`, since there is no earlier transaction to aggregate.

On this dataset, the leakage fix had a small effect on the headline number, which is itself informative:

<div align="center">

| Metric | Whole-dataset aggregation (leaked) | Point-in-time (current) |
|---|---|---|
| AUC | 0.913 | 0.915 |

</div>

> A small change here does not mean leakage never matters, it means this dataset's fraud signal happens to be recoverable from point-in-time features too. The fix still matters for correctness: the leaked numbers were never achievable by a system scoring transactions as they arrive.

---

## Model Output

Running the pipeline produces a per-transaction risk score and a per-user rollup:

<div align="center">

| Output | Contents |
|---|---|
| `fraud_scores.csv` | Every transaction, ranked by `fraud_proba` (highest risk first), with `tx_id`, `user_id`, `amount`, and the true `label` |
| `fraud_summary.csv` | The top 200 highest-risk transactions, aggregated per user: `max_fraud_proba` and `total_amount` |
| `metrics.json` | `auc`, `precision`, `recall`, `f1` on the holdout set at `--threshold` |

</div>

These are ranked outputs for human review, not automated accept/reject decisions.

---

## Model Artifacts and Loading Safety

If `--save-model` is given, the fitted scaler, classifier, and feature list are stored together as a `joblib` file. **Deserializing a pickle (which `joblib` uses under the hood) executes arbitrary code**, so only load a model file you produced yourself or fully trust, never one downloaded from an untrusted source.

There is currently no checksum sidecar or integrity check on load; treat any `--save-model` output the same way you would treat any other pickled Python object.

---

## Evaluation Metrics

Evaluation uses a stratified train/test split (`test_size=0.25` by default) with `class_weight="balanced"` to account for severe class imbalance.

<div align="center">

| Metric | Why it matters |
|---|---|
| AUC | Ranking quality across thresholds, independent of the chosen cutoff |
| Precision | Share of flagged transactions that are actually fraud |
| Recall | Share of actual fraud transactions that get flagged |
| F1 | Balance between precision and recall at the chosen threshold |

</div>

Example results from the included run (67,336 transactions: 625 fraud, 66,711 legitimate), at the default `--threshold 0.5`:

<div align="center">

| Metric | Logistic Regression (default) | Random Forest (`--model random_forest`) |
|---|---|---|
| AUC | 0.915 | 0.907 |
| Precision | 0.375 | 0.461 |
| Recall | 0.821 | 0.263 |
| F1 | 0.515 | 0.335 |

</div>

> Precision around 0.38–0.46 means most flagged transactions are still false positives, expected at this level of class imbalance (about 1 fraud per 107 legitimate transactions), but worth knowing before treating either model's output as a decision rather than a ranking to review.

---

## Visual Reports

<div align="center">

<img width="500" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/db3669b0-0372-47d8-a3bf-08584dd9e94b" />

**Analysis:** The ROC curve shows the trade-off between true positive rate (recall) and false positive rate for the default Logistic Regression model on the holdout set. A curve closer to the top-left corner indicates stronger ranking performance; the diagonal reference line is what random guessing would produce.

</div>

---

## Testing and CI

Run unit tests locally:

```bash
pytest -q
```

Run the full quality gate (same thing CI runs):

```bash
make gate
```

This runs, in order: `ruff check`, `black --check`, `mypy src`, and `pytest` (coverage must stay at or above 90%).

The GitHub Actions workflow checks:

- dependency installation (pinned runtime and dev requirements)
- linting with Ruff
- format checking with Black
- type checking with mypy
- unit tests with a 90%-minimum coverage gate
- a full pipeline run end-to-end on the real dataset
- upload of `outputs/` as a build artifact

CI is defined in:

```text
.github/workflows/ci.yml
```

Dependabot (`.github/dependabot.yml`) opens weekly PRs to keep pinned pip and GitHub Actions dependencies current. Optional: `pre-commit install` runs ruff/black/mypy automatically on each commit, using the same versions as CI.

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/create_db.py` | Loads and validates the labeled CSV into a SQLite table |
| `src/queries.sql` | Point-in-time SQL feature engineering via window functions |
| `src/train_supervised.py` | Runs the SQL query, trains/evaluates the model, writes artifacts, CLI |
| `src/utils.py` | Shared I/O and ROC-plotting helpers |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest/coverage) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- **Small SQLite/logistic-regression pipeline**, not a production fraud system, no online scoring, no model registry, no drift monitoring
- **Class imbalance is real** (625 fraud / 66,711 legitimate transactions); `class_weight="balanced"` helps, but precision stays well below 1.0 at any reasonable recall
- The point-in-time feature fix matters for correctness even though its effect on this dataset's headline AUC was small
- The Random Forest option is provided for comparison, not because it has been shown to generalize better on unseen data
- No cross-validation or hyperparameter search is performed; a single fixed train/test split is used

The project is strongest as a portfolio demonstration of honest, leakage-aware feature engineering and evaluation, combining SQL and Python.

---

## Responsible Use

This repository is intended for:

- machine learning and SQL feature-engineering education
- demonstrating honest, leakage-aware evaluation
- practicing a SQL-plus-Python classification workflow
- responsible-ML documentation practice
- portfolio demonstration

It should not be used as-is for:

- automated transaction blocking or account actions
- credit, risk, or compliance decisions
- production fraud operations without human review
- any high-stakes financial decision

Any real deployment would require richer features, online scoring infrastructure, a model registry, drift monitoring, fairness review, and a human escalation process.

---

## Future Improvements

Potential next improvements:

- Add cross-validation and hyperparameter tuning instead of a single fixed split
- Add calibration metrics (e.g., Brier score, reliability plots) alongside AUC/precision/recall
- Add feature-importance or SHAP explanations for the trained model
- Add a checksum sidecar for saved model artifacts
- Package the pipeline as a small online-scoring service
- Explore gradient-boosted models (e.g., XGBoost, LightGBM) alongside Logistic Regression and Random Forest
- Add drift monitoring between training and scoring data

---

## Tech Stack

- Python
- SQLite
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib
- pytest
- Ruff
- Black
- mypy
- GitHub Actions
- Dependabot

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License (see `LICENSE`).

It is intended for educational, research, and portfolio purposes. If you use or modify this project, please keep the responsible-use notes and limitations clear.
