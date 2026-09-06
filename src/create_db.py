#!/usr/bin/env python3
"""Load a labeled transactions CSV into a SQLite database."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

TABLE_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS transactions ("
    "tx_id INTEGER, user_id TEXT, date TEXT, region TEXT, "
    "merchant TEXT, amount REAL, label INTEGER)"
)


def load_csv_to_db(csv_path: str | Path, db_path: str | Path) -> int:
    """Load `csv_path` into a `transactions` table in `db_path`.

    Returns the number of rows loaded. Raises FileNotFoundError if
    `csv_path` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as con:
        con.execute(TABLE_SCHEMA)
        df.to_sql("transactions", con, if_exists="replace", index=False)
    return len(df)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load a labeled transactions CSV into SQLite")
    ap.add_argument(
        "--csv",
        required=os.environ.get("FRAUD_CSV_PATH") is None,
        default=os.environ.get("FRAUD_CSV_PATH"),
        help="Path to the transactions CSV (env: FRAUD_CSV_PATH)",
    )
    ap.add_argument(
        "--db",
        default=os.environ.get("FRAUD_DB_PATH", "fraud.db"),
        help="Path to the SQLite database to create (env: FRAUD_DB_PATH)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n_rows = load_csv_to_db(args.csv, args.db)
    print(f"Loaded {n_rows} rows: {args.csv} -> {args.db}")


if __name__ == "__main__":
    main()
