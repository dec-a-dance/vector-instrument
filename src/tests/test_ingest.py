import json
import sqlite3

import pandas as pd
import pytest

from app.ingest import ingest_file
from app.storage import Storage


def test_ingest_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30]
    })
    df.to_csv(csv_path, index=False)

    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))

    result = ingest_file(str(csv_path), storage)

    assert result["rows_inserted"] == 3
    assert result["dimensions"] == 2
    assert result["source_type"] == "csv"
    assert storage.count_vectors() == 3


def test_ingest_json_list(tmp_path):
    json_path = tmp_path / "data.json"
    data = [
        {"x": 1, "y": 2},
        {"x": 2, "y": 3}
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))

    result = ingest_file(str(json_path), storage)

    assert result["rows_inserted"] == 2
    assert result["dimensions"] == 2
    assert result["source_type"] == "json"


def test_ingest_sqlite(tmp_path):
    sqlite_path = tmp_path / "input.db"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE vectors_in (a REAL, b REAL)")
    conn.execute("INSERT INTO vectors_in (a, b) VALUES (1, 10)")
    conn.execute("INSERT INTO vectors_in (a, b) VALUES (2, 20)")
    conn.commit()
    conn.close()

    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))

    result = ingest_file(str(sqlite_path), storage, "vectors_in")

    assert result["rows_inserted"] == 2
    assert result["dimensions"] == 2
    assert result["source_type"] == "sqlite"


def test_ingest_invalid_table_name(tmp_path):
    sqlite_path = tmp_path / "input.db"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE safe_table (a REAL)")
    conn.commit()
    conn.close()

    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))

    with pytest.raises(ValueError, match="Invalid table name"):
        ingest_file(str(sqlite_path), storage, "safe_table; DROP TABLE safe_table;")