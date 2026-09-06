import sqlite3

import pandas as pd
import pytest

from create_db import load_csv_to_db


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
