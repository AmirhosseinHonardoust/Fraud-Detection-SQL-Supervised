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

REQUIRED_COLUMNS = ["tx_id", "user_id", "date", "region", "merchant", "amount", "label"]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` is missing any column `queries.sql` expects.

    Catches malformed/mismatched CSVs at load time instead of failing later
    with a confusing SQL or training error.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def load_csv_to_db(csv_path: str | Path, db_path: str | Path) -> int:
    """Load `csv_path` into a `transactions` table in `db_path`.

    Returns the number of rows loaded. Raises FileNotFoundError if
    `csv_path` does not exist, or ValueError if it's missing a required
    column.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df)
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
