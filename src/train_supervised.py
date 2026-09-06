#!/usr/bin/env python3
"""Train a supervised (Logistic Regression) fraud model on SQL-engineered features."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure `utils` resolves to the module next to this file regardless of the
# current working directory or how this script is invoked (direct run, pytest
# collection, etc).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_outdir, plot_roc, save_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["amount", "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount"]

MODEL_CHOICES = ("logreg", "random_forest")


def _build_model(model: str, random_state: int) -> ClassifierMixin:
    """Return an unfitted classifier for `model` (see MODEL_CHOICES).

    `logreg` (the default) reproduces the exact estimator this project has
    always used. `random_forest` is an opt-in alternative for comparison;
    it is not a drop-in behavioral replacement, so it's never used unless
    explicitly requested via `--model random_forest`.
    """
    if model == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        )
    raise ValueError(f"Unknown model {model!r}; choose one of {MODEL_CHOICES}")


def _strip_sql_line_comments(sql_text: str) -> str:
    """Strip `-- ...` line comments before statement-splitting on `;`.

    Only comments are handled here (a `--` outside a single-quoted string
    truncates the rest of its line); a semicolon inside a string literal is
    still not supported, since a full SQL tokenizer is out of scope for this
    project's small, hand-authored `queries.sql`. This is enough to stop a
    semicolon that merely appears in a *comment* from being mistaken for a
    statement separator.
    """
    out_lines = []
    for line in sql_text.splitlines():
        in_string = False
        cut = len(line)
        i = 0
        while i < len(line) - 1:
            if line[i] == "'":
                in_string = not in_string
            elif not in_string and line[i : i + 2] == "--":
                cut = i
                break
            i += 1
        out_lines.append(line[:cut])
    return "\n".join(out_lines)


def _threshold_type(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"--threshold must be between 0.0 and 1.0, got {parsed}")
    return parsed


def _test_size_type(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            f"--test-size must be strictly between 0.0 and 1.0, got {parsed}"
        )
    return parsed


def run_training(
    db_path: str | Path,
    sql_path: str | Path,
    outdir: str | Path,
    *,
    threshold: float = 0.5,
    test_size: float = 0.25,
    random_state: int = 42,
    save_model: str | Path | None = None,
    model: str = "logreg",
) -> None:
    """Run the SQL feature engineering + supervised training pipeline.

    `threshold`, `test_size`, and `random_state` tune the split/classification
    decision without touching code. If `save_model` is given, the fitted
    scaler + classifier + feature list are persisted there via joblib so a
    later process can load them and score new transactions without retraining.
    `model` selects the classifier (see MODEL_CHOICES); it defaults to
    `"logreg"`, which reproduces this project's original behavior exactly.
    """
    if model not in MODEL_CHOICES:
        raise ValueError(f"Unknown model {model!r}; choose one of {MODEL_CHOICES}")
    db_path = Path(db_path)
    sql_path = Path(sql_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(Path(outdir) / "charts")

    # Read SQL and split into statements (setup views/CTEs + final SELECT).
    sql_text = _strip_sql_line_comments(sql_path.read_text(encoding="utf-8"))
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    if not statements:
        raise RuntimeError("No SQL statements found in queries.sql")

    setup_script = ";\n".join(statements[:-1]) + (";" if len(statements) > 1 else "")
    final_select = statements[-1]

    # Execute SQL: setup first, then final SELECT.
    with sqlite3.connect(db_path) as con:
        if setup_script:
            con.executescript(setup_script)
        df = pd.read_sql_query(final_select, con)

    if df.empty:
        raise RuntimeError("Final SELECT returned no rows. Check your data and SQL.")

    # Supervised labels.
    if "label" not in df.columns:
        raise RuntimeError("Expected 'label' column in SQL output for supervised training.")
    df = df.dropna(subset=["label"])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"SQL output is missing expected feature column(s): {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df["label"].astype(int)

    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale + train.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = _build_model(model, random_state)
    clf.fit(X_train_s, y_train)

    # Evaluate.
    proba = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, (proba > threshold).astype(int), average="binary", zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_test, proba)
    plot_roc(fpr, tpr, charts_dir / "roc_curve.png")

    metrics = {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    with open(Path(outdir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", metrics)

    # Rank all transactions by predicted risk.
    full_s = scaler.transform(X)
    all_proba = clf.predict_proba(full_s)[:, 1]
    ranked = df.copy()
    ranked["fraud_proba"] = all_proba
    ranked = ranked.sort_values("fraud_proba", ascending=False)

    save_csv(
        ranked[["tx_id", "user_id", "amount", "fraud_proba", "label"]],
        Path(outdir) / "fraud_scores.csv",
    )
    summary = (
        ranked.head(200)
        .groupby("user_id")
        .agg(max_fraud_proba=("fraud_proba", "max"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values(["max_fraud_proba", "total_amount"], ascending=False)
    )
    save_csv(summary, Path(outdir) / "fraud_summary.csv")

    if save_model is not None:
        save_model = Path(save_model)
        save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": clf, "scaler": scaler, "feature_cols": FEATURE_COLS},
            save_model,
        )
        logger.info("Model saved to: %s", save_model.resolve())

    logger.info("Artifacts saved to: %s", Path(outdir).resolve())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Supervised fraud detection (Logistic Regression) with SQL features"
    )
    ap.add_argument(
        "--db",
        default=os.environ.get("FRAUD_DB_PATH", "fraud.db"),
        help="Path to the SQLite database (env: FRAUD_DB_PATH)",
    )
    ap.add_argument(
        "--sql",
        default=os.environ.get("FRAUD_SQL_PATH", "src/queries.sql"),
        help="Path to the feature-engineering SQL file (env: FRAUD_SQL_PATH)",
    )
    ap.add_argument(
        "--outdir",
        default=os.environ.get("FRAUD_OUTDIR", "outputs"),
        help="Directory to write metrics/scores/charts to (env: FRAUD_OUTDIR)",
    )
    ap.add_argument(
        "--threshold",
        type=_threshold_type,
        default=os.environ.get("FRAUD_THRESHOLD", "0.5"),
        help="Probability threshold for the fraud/not-fraud decision, in [0, 1] "
        "(env: FRAUD_THRESHOLD)",
    )
    ap.add_argument(
        "--test-size",
        type=_test_size_type,
        default=os.environ.get("FRAUD_TEST_SIZE", "0.25"),
        help="Fraction of data held out for evaluation, in (0, 1) (env: FRAUD_TEST_SIZE)",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=int(os.environ.get("FRAUD_RANDOM_STATE", "42")),
        help="Random seed for the train/test split (env: FRAUD_RANDOM_STATE)",
    )
    ap.add_argument(
        "--save-model",
        default=os.environ.get("FRAUD_MODEL_PATH"),
        help="If set, path to persist the fitted scaler+model via joblib (env: FRAUD_MODEL_PATH)",
    )
    ap.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=os.environ.get("FRAUD_MODEL", "logreg"),
        help="Classifier to train (env: FRAUD_MODEL). Default 'logreg' matches "
        "this project's original behavior; 'random_forest' is an opt-in alternative "
        "for comparison.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        args.db,
        args.sql,
        args.outdir,
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        save_model=args.save_model,
        model=args.model,
    )


if __name__ == "__main__":
    main()
