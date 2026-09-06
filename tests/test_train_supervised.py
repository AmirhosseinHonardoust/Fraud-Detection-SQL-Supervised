import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import train_supervised
from create_db import load_csv_to_db
from train_supervised import parse_args, run_training

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


def test_run_training_default_params_match_hardcoded_baseline(tmp_path):
    """Passing no keyword args must reproduce the exact metrics the old
    hardcoded threshold=0.5/test_size=0.25/random_state=42 produced."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    outdir_default = tmp_path / "out_default"
    outdir_explicit = tmp_path / "out_explicit"
    run_training(db_path, QUERIES_SQL, outdir_default)
    run_training(
        db_path,
        QUERIES_SQL,
        outdir_explicit,
        threshold=0.5,
        test_size=0.25,
        random_state=42,
    )

    metrics_default = json.loads((outdir_default / "metrics.json").read_text())
    metrics_explicit = json.loads((outdir_explicit / "metrics.json").read_text())
    assert metrics_default == metrics_explicit


def test_run_training_random_state_is_reproducible(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    run_training(db_path, QUERIES_SQL, out_a, random_state=7)
    run_training(db_path, QUERIES_SQL, out_b, random_state=7)

    assert json.loads((out_a / "metrics.json").read_text()) == json.loads(
        (out_b / "metrics.json").read_text()
    )


def test_run_training_threshold_changes_precision_recall(tmp_path):
    """A near-0 threshold should flag ~everyone as fraud: recall -> 1.0."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    outdir = tmp_path / "out"
    run_training(db_path, QUERIES_SQL, outdir, threshold=0.0)

    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["recall"] == 1.0


def test_run_training_save_model_persists_scaler_and_classifier(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    model_path = tmp_path / "model" / "fraud_model.joblib"
    run_training(db_path, QUERIES_SQL, tmp_path / "out", save_model=model_path)

    assert model_path.is_file()
    bundle = joblib.load(model_path)
    assert set(bundle) == {"model", "scaler", "feature_cols"}
    assert bundle["feature_cols"] == train_supervised.FEATURE_COLS

    # The loaded scaler + model should reproduce a valid probability for a
    # feature row shaped like the training data.
    row = pd.DataFrame([[10.0, 0, 0.0, 0.0, 0, 0.0]], columns=bundle["feature_cols"])
    scaled = bundle["scaler"].transform(row)
    proba = bundle["model"].predict_proba(scaled)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


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


def test_run_training_missing_feature_column_raises(tmp_path):
    """If queries.sql drifts and stops producing a required feature column,
    run_training should fail loudly instead of silently training on fewer
    features."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    broken_sql = tmp_path / "broken.sql"
    broken_sql.write_text("SELECT tx_id, user_id, amount, label FROM transactions;")

    with pytest.raises(RuntimeError, match="missing expected feature column"):
        run_training(db_path, broken_sql, tmp_path / "out")


def test_run_training_empty_sql_raises(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    empty_sql = tmp_path / "empty.sql"
    empty_sql.write_text("   ;  ")

    with pytest.raises(RuntimeError, match="No SQL statements"):
        run_training(db_path, empty_sql, tmp_path / "out")


def test_run_training_empty_result_raises(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    no_rows_sql = tmp_path / "no_rows.sql"
    no_rows_sql.write_text("SELECT * FROM transactions WHERE 1 = 0;")

    with pytest.raises(RuntimeError, match="no rows"):
        run_training(db_path, no_rows_sql, tmp_path / "out")


def test_run_training_missing_label_raises(tmp_path):
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    no_label_sql = tmp_path / "no_label.sql"
    no_label_sql.write_text("SELECT tx_id, user_id, amount FROM transactions;")

    with pytest.raises(RuntimeError, match="Expected 'label' column"):
        run_training(db_path, no_label_sql, tmp_path / "out")


def test_run_training_with_setup_statements(tmp_path):
    """Exercise the multi-statement (setup script + final SELECT) branch,
    which the single-statement production queries.sql doesn't hit."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)

    multi_sql = tmp_path / "multi.sql"
    multi_sql.write_text("""
        CREATE TEMP VIEW user_stats AS
        SELECT user_id, COUNT(*) AS tx_count, AVG(amount) AS avg_amount,
               SUM(amount) AS total_amount
        FROM transactions GROUP BY user_id;
        SELECT t.tx_id, t.user_id, t.amount, us.tx_count, us.avg_amount,
               us.total_amount, 0 AS daily_tx, 0.0 AS daily_amount, t.label
        FROM transactions t JOIN user_stats us ON t.user_id = us.user_id;
        """)

    run_training(db_path, multi_sql, tmp_path / "out")

    assert (tmp_path / "out" / "metrics.json").is_file()


def test_parse_args_env_var_defaults(monkeypatch):
    monkeypatch.setenv("FRAUD_DB_PATH", "env.db")
    monkeypatch.setenv("FRAUD_SQL_PATH", "env.sql")
    monkeypatch.setenv("FRAUD_OUTDIR", "env_out")
    monkeypatch.setattr(sys, "argv", ["train_supervised.py"])

    args = parse_args()

    assert args.db == "env.db"
    assert args.sql == "env.sql"
    assert args.outdir == "env_out"


def test_parse_args_defaults_without_env(monkeypatch):
    for var in (
        "FRAUD_THRESHOLD",
        "FRAUD_TEST_SIZE",
        "FRAUD_RANDOM_STATE",
        "FRAUD_MODEL_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_supervised.py"])

    args = parse_args()

    assert args.threshold == 0.5
    assert args.test_size == 0.25
    assert args.random_state == 42
    assert args.save_model is None


def test_parse_args_cli_flags_override_env(monkeypatch):
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.5")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_supervised.py",
            "--threshold",
            "0.7",
            "--test-size",
            "0.3",
            "--random-state",
            "123",
            "--save-model",
            "model.joblib",
        ],
    )

    args = parse_args()

    assert args.threshold == 0.7
    assert args.test_size == 0.3
    assert args.random_state == 123
    assert args.save_model == "model.joblib"


def test_cli_end_to_end(tmp_path):
    """Smoke-test the actual command line entry points end-to-end."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "create_db.py"),
            "--csv",
            str(csv_path),
            "--db",
            str(db_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    outdir = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "train_supervised.py"),
            "--db",
            str(db_path),
            "--sql",
            str(QUERIES_SQL),
            "--outdir",
            str(outdir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (outdir / "metrics.json").is_file()
    assert (outdir / "fraud_scores.csv").is_file()
    assert (outdir / "charts" / "roc_curve.png").is_file()


def test_main_in_process(tmp_path, monkeypatch):
    """Exercise train_supervised.main() in-process so it's measured by
    coverage, unlike the subprocess-based CLI test above."""
    csv_path = tmp_path / "tx.csv"
    _make_synthetic_csv(csv_path)
    db_path = tmp_path / "fraud.db"
    load_csv_to_db(csv_path, db_path)
    outdir = tmp_path / "outputs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_supervised.py",
            "--db",
            str(db_path),
            "--sql",
            str(QUERIES_SQL),
            "--outdir",
            str(outdir),
        ],
    )
    train_supervised.main()

    assert (outdir / "metrics.json").is_file()
