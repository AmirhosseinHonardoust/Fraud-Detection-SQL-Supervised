# Fraud Detection [SQL + Python (Supervised)]

[![CI](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Supervised/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Supervised/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict fraudulent transactions using **SQL (SQLite)** for feature engineering and **Python** with Logistic Regression for supervised classification.

---

## Overview

This project extends the unsupervised version by introducing **labeled data** and **supervised learning**.  
It demonstrates a complete fraud prediction pipeline, from SQL feature generation to model training, evaluation, and visualization.

---

## Workflow

1. **Load labeled data into SQLite**
2. **Run SQL feature engineering**
   - Compute per-user and daily transaction statistics
3. **Train Logistic Regression model**
   - Input: engineered SQL features
   - Output: fraud probability for each transaction
4. **Evaluate model performance**
   - AUC, Precision, Recall, F1-score
5. **Visualize ROC curve**

---

## Project Structure

```
fraud-detection-sql-supervised/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ transactions_labeled.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ train_supervised.py
│  └─ utils.py
└─ outputs/
   ├─ metrics.json
   ├─ fraud_scores.csv
   ├─ fraud_summary.csv
   └─ charts/
       └─ roc_curve.png
```

---

## Dataset Schema

| Column | Description |
|---------|--------------|
| tx_id | Transaction ID |
| user_id | Unique user identifier |
| date | Transaction date |
| region | User region |
| merchant | Merchant name |
| amount | Transaction amount |
| label | 1 = Fraudulent, 0 = Legitimate |

---

## SQL Feature Engineering

Every aggregate feature is **point-in-time**: it's computed only from a user's
transactions *strictly before* the current one (via SQLite window functions
with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`). A transaction never
sees its own amount or anything that happens later, which matches how a real
fraud-scoring system would see data as it arrives.

```sql
SELECT
  t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
  COALESCE(COUNT(t.amount) OVER user_hist, 0) AS tx_count,
  COALESCE(AVG(t.amount) OVER user_hist, 0.0) AS avg_amount,
  COALESCE(SUM(t.amount) OVER user_hist, 0.0) AS total_amount,
  COALESCE(COUNT(t.amount) OVER daily_hist, 0) AS daily_tx,
  COALESCE(SUM(t.amount) OVER daily_hist, 0.0) AS daily_amount,
  t.label
FROM transactions t
WINDOW
  user_hist AS (
    PARTITION BY t.user_id ORDER BY t.date, t.tx_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ),
  daily_hist AS (
    PARTITION BY t.user_id, t.date ORDER BY t.tx_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  )
ORDER BY t.date, t.tx_id;
```

An earlier version of this query aggregated over the *entire* dataset
(including future transactions), which leaked information a live system
would not have yet. See [Limitations](#limitations) for the measured impact
of fixing this.

---

## Machine Learning

Model: **Logistic Regression**

- Trained on labeled transaction data  
- Balanced class weights for rare fraud cases  
- Evaluated using ROC AUC, precision, recall, and F1-score  
- Generates probability scores (`fraud_proba`) for each transaction

---

## Visualization

### ROC Curve
<img width="900" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/db3669b0-0372-47d8-a3bf-08584dd9e94b" />

The ROC curve shows the trade-off between true positive rate (recall) and false positive rate.  
A curve closer to the top-left corner indicates stronger predictive performance.

---

## Tools & Libraries

| Tool | Purpose |
|------|----------|
| **SQLite** | Data storage and feature generation |
| **Python** | ML training and evaluation |
| **pandas** | Data handling |
| **scikit-learn** | Model building and metrics |
| **matplotlib** | Visualization |

Exact versions are pinned in `requirements.txt` (runtime) and
`requirements-dev.txt` (lint/type-check/test tooling) — these are the versions
CI installs and the pipeline is verified against.

---

## Usage

### Load Data into SQLite
```bash
python src/create_db.py --csv data/transactions_labeled.csv --db fraud.db
```

### Train and Evaluate Model
```bash
python src/train_supervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

Every path can also be set via environment variable instead of a flag:
`FRAUD_CSV_PATH`, `FRAUD_DB_PATH`, `FRAUD_SQL_PATH`, `FRAUD_OUTDIR`. A flag
always overrides the matching env var.

`train_supervised.py` also accepts:

| Flag | Env var | Default | Purpose |
|------|---------|---------|---------|
| `--threshold` | `FRAUD_THRESHOLD` | `0.5` | Probability cutoff for the fraud/not-fraud decision used in precision/recall/F1 |
| `--test-size` | `FRAUD_TEST_SIZE` | `0.25` | Fraction of data held out for evaluation |
| `--random-state` | `FRAUD_RANDOM_STATE` | `42` | Seed for the train/test split |
| `--save-model` | `FRAUD_MODEL_PATH` | *(unset)* | If given, persists the fitted scaler + classifier + feature list to this path via `joblib`, so a later process can load it and score new transactions without retraining |

---

## Outputs

Running the two commands above writes these files to `--outdir` (default
`outputs/`). They are **not** committed to the repo — regenerate them locally,
or download them from the `pipeline-outputs` artifact on any CI run.

| File | Description |
|------|--------------|
| `metrics.json` | Model performance metrics |
| `fraud_scores.csv` | Ranked transactions with fraud probability |
| `fraud_summary.csv` | Aggregated user-level fraud summary |
| `charts/roc_curve.png` | ROC curve visualization |

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check src tests
black --check src tests
mypy src
pytest -q          # runs with coverage, fails under 90%
```

Or run the whole gate in one command: `make gate` (see `Makefile`; `make run`
runs the pipeline end-to-end). See `CONTRIBUTING.md` for the full workflow.

Optional: `pre-commit install` runs ruff/black/mypy automatically on each
commit, using the same versions as CI (`.pre-commit-config.yaml`).

CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull
request, then executes the full pipeline and uploads `outputs/` as a build
artifact. Dependabot (`.github/dependabot.yml`) opens weekly PRs to keep
pinned pip and GitHub Actions dependencies current.

## Limitations

- **Small SQLite/logistic-regression pipeline**, not a production fraud
  system. There's no online scoring, no model registry, no drift monitoring.
- **Point-in-time features** (see above) replaced an earlier version that
  aggregated over the whole dataset. On this dataset the practical effect on
  reported metrics was small (AUC 0.913 → 0.915), but the fix matters for
  correctness: the previous numbers would not have been achievable in a real
  system scoring transactions as they arrive.
- **Class imbalance** is real (625 fraud / 66,711 legitimate transactions).
  `class_weight="balanced"` helps, but precision (~0.38) means most flagged
  transactions are still false positives — expected at this level of
  imbalance, but worth knowing before treating the scores as decisions rather
  than a ranking to review.

## Conclusion

This project demonstrates a complete **supervised fraud detection workflow** using SQL and Python.  
It combines data engineering, model training, and evaluation into a single reproducible pipeline suitable for production-ready analytics and portfolio demonstration.
