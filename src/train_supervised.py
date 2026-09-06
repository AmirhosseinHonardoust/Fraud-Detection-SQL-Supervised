#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
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


def run_training(db_path: str | Path, sql_path: str | Path, outdir: str | Path) -> None:
    db_path = Path(db_path)
    sql_path = Path(sql_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(Path(outdir) / "charts")

    # Read SQL and split into statements (setup views/CTEs + final SELECT).
    sql_text = sql_path.read_text(encoding="utf-8")
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
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale + train.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    # Evaluate.
    proba = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, (proba > 0.5).astype(int), average="binary", zero_division=0
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

    logger.info("Artifacts saved to: %s", Path(outdir).resolve())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Supervised fraud detection (Logistic Regression) with SQL features"
    )
    ap.add_argument("--db", default="fraud.db")
    ap.add_argument("--sql", default="src/queries.sql")
    ap.add_argument("--outdir", default="outputs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.db, args.sql, args.outdir)


if __name__ == "__main__":
    main()
