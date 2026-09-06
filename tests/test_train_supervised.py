import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from create_db import load_csv_to_db
from train_supervised import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_SQL = REPO_ROOT / "src" / "queries.sql"


def _make_synthetic_csv(path: Path, n_users: int = 8, n_days: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    tx_id = 1
    for day in range(n_days):
        date = f"2024-01-{day + 1:02d}"
        for user in range(n_users):
            if rng.random() < 0.7:  # not every user transacts every day
                continue
            amount = float(rng.uniform(10, 200))
            label = int(rng.random() < 0.1)
            rows.append(
                {
                    "tx_id": tx_id,
                    "user_id": f"U{user}",
                    "date": date,
                    "region": "North",
                    "merchant": "StoreA",
                    "amount": amount,
                    "label": label,
                }
            )
            tx_id += 1
    df = pd.DataFrame(rows)
    # Guarantee both classes exist so stratified split works.
    df.loc[0, "label"] = 0
    df.loc[1, "label"] = 1
    df.loc[2, "label"] = 0
    df.loc[3, "label"] = 1
    df.to_csv(path, index=False)


def test_run_training_end_to_end(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    outdir = tmp_path / "outputs"
    run_training(db_path, QUERIES_SQL, outdir)

    metrics = json.loads((outdir / "metrics.json").read_text())
    for key in ("auc", "precision", "recall", "f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    scores = pd.read_csv(outdir / "fraud_scores.csv")
    n_input_rows = len(pd.read_csv(csv_path))
    assert len(scores) == n_input_rows
    assert (outdir / "charts" / "roc_curve.png").is_file()
    assert (outdir / "fraud_summary.csv").is_file()


def test_run_training_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_training(tmp_path / "missing.db", QUERIES_SQL, tmp_path / "out")


def test_run_training_missing_sql_raises(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    with pytest.raises(FileNotFoundError):
        run_training(db_path, tmp_path / "missing.sql", tmp_path / "out")


def test_queries_sql_features_are_point_in_time(tmp_path):
    """Regression test for the look-ahead leakage fix: every user's first
    transaction (by date, tx_id) must show tx_count == 0, since there is no
    earlier transaction to aggregate."""
    import sqlite3

    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    sql = QUERIES_SQL.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con)

    first_per_user = df.sort_values(["user_id", "date", "tx_id"]).groupby("user_id").first()
    assert (first_per_user["tx_count"] == 0).all()
