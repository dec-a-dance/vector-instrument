import json
import re
import sqlite3
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .storage import Storage


def _validate_table_name(table: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("Invalid table name")


def _load_dataframe(file_path: str, table_name: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        return df, "csv"

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            if "data" in raw and isinstance(raw["data"], list):
                df = pd.DataFrame(raw["data"])
            else:
                df = pd.DataFrame([raw])
        else:
            raise ValueError("Unsupported JSON structure")

        return df, "json"

    if suffix in {".sqlite", ".db"}:
        if not table_name:
            raise ValueError("table_name is required for SQLite input")
        _validate_table_name(table_name)

        conn = sqlite3.connect(path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            return df, "sqlite"
        finally:
            conn.close()

    raise ValueError("Unsupported file type. Use CSV, JSON, or SQLite")


def _normalize_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if numeric_df.empty:
        raise ValueError("No numeric columns found in input data")

    if numeric_df.isnull().any().any():
        numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))

    values = numeric_df.to_numpy(dtype=float)

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Input data is empty")

    min_v = values.min(axis=0)
    max_v = values.max(axis=0)
    denom = max_v - min_v
    denom[denom == 0.0] = 1.0

    normalized = (values - min_v) / denom
    return normalized, list(numeric_df.columns)


def ingest_file(
    file_path: str,
    storage: Storage,
    table_name: Optional[str] = None,
    keep_columns: Optional[Sequence[str]] = None,
) -> dict:
    df, source_type = _load_dataframe(file_path, table_name)
    keep_columns = list(keep_columns or [])

    missing_keep_columns = [c for c in keep_columns if c not in df.columns]
    if missing_keep_columns:
        raise ValueError(f"Unknown keep_columns: {missing_keep_columns}")

    numeric_df = df.select_dtypes(include=["number"]).copy()

    if numeric_df.empty:
        raise ValueError("No numeric columns found in input data")

    normalized, used_feature_columns = _normalize_dataframe(numeric_df)

    payloads = None
    if keep_columns:
        kept_df = df[keep_columns].copy()
        payloads = kept_df.to_dict(orient="records")

    storage.clear_vectors()
    storage.insert_vectors(normalized.tolist(), payloads=payloads)

    storage.set_metadata("dimensions", json.dumps(int(normalized.shape[1])))
    storage.set_metadata("source_type", source_type)
    storage.set_metadata("feature_columns", json.dumps(used_feature_columns))
    storage.set_metadata("keep_columns", json.dumps(keep_columns))

    return {
        "rows_inserted": int(normalized.shape[0]),
        "dimensions": int(normalized.shape[1]),
        "source_type": source_type,
        "feature_columns": used_feature_columns,
        "keep_columns": keep_columns,
    }