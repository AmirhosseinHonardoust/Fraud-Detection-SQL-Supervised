import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import create_db
from create_db import load_csv_to_db, parse_args

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_csv_to_db_roundtrip(tmp_path):
    csv_path = tmp_path / "tx.csv"
    df = pd.DataFrame(
        {
            "tx_id": [1, 2],
            "user_id": ["U1", "U2"],
            "date": ["2024-01-01", "2024-01-02"],
            "region": ["North", "South"],
            "merchant": ["StoreA", "StoreB"],
            "amount": [10.0, 20.0],
            "label": [0, 1],
        }
    )
    df.to_csv(csv_path, index=False)
    db_path = tmp_path / "test.db"

    n_rows = load_csv_to_db(csv_path, db_path)

    assert n_rows == 2
    with sqlite3.connect(db_path) as con:
        loaded = pd.read_sql_query("SELECT * FROM transactions ORDER BY tx_id", con)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_csv_to_db_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_to_db(tmp_path / "does_not_exist.csv", tmp_path / "out.db")


def test_parse_args_env_var_defaults(monkeypatch):
    monkeypatch.setenv("FRAUD_CSV_PATH", "data/env_tx.csv")
    monkeypatch.setenv("FRAUD_DB_PATH", "env_fraud.db")
    monkeypatch.setattr(sys, "argv", ["create_db.py"])

    args = parse_args()

    assert args.csv == "data/env_tx.csv"
    assert args.db == "env_fraud.db"


def test_parse_args_requires_csv_without_env(monkeypatch):
    monkeypatch.delenv("FRAUD_CSV_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["create_db.py"])

    with pytest.raises(SystemExit):
        parse_args()


def test_cli_end_to_end(tmp_path):
    """Smoke-test the actual command line entry point, not just the function."""
    csv_path = tmp_path / "tx.csv"
    pd.DataFrame(
        {
            "tx_id": [1],
            "user_id": ["U1"],
            "date": ["2024-01-01"],
            "region": ["North"],
            "merchant": ["StoreA"],
            "amount": [10.0],
            "label": [0],
        }
    ).to_csv(csv_path, index=False)
    db_path = tmp_path / "cli.db"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "create_db.py"),
            "--csv",
            str(csv_path),
            "--db",
            str(db_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert db_path.is_file()
    assert "Loaded 1 rows" in result.stdout


def test_main_in_process(tmp_path, monkeypatch, capsys):
    """Exercise create_db.main() in-process (covered by coverage, unlike the
    subprocess-based CLI test above which runs in a separate interpreter)."""
    csv_path = tmp_path / "tx.csv"
    pd.DataFrame(
        {
            "tx_id": [1, 2],
            "user_id": ["U1", "U1"],
            "date": ["2024-01-01", "2024-01-02"],
            "region": ["North", "North"],
            "merchant": ["StoreA", "StoreA"],
            "amount": [10.0, 20.0],
            "label": [0, 1],
        }
    ).to_csv(csv_path, index=False)
    db_path = tmp_path / "main.db"

    monkeypatch.setattr(sys, "argv", ["create_db.py", "--csv", str(csv_path), "--db", str(db_path)])
    create_db.main()

    assert db_path.is_file()
    assert "Loaded 2 rows" in capsys.readouterr().out
