from typing import List, Optional, Dict, Any

import numpy as np

from .storage import Storage


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def knn_search(
    storage: Storage,
    query_vector: List[float],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[dict]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dataset = storage.fetch_vectors_with_payloads(filters=filters)
    if not dataset:
        return []

    q = np.array(query_vector, dtype=float)
    if q.ndim != 1:
        raise ValueError("Query vector must be one-dimensional")

    q = _l2_normalize(q)

    rows = []
    for item in dataset:
        vec = item["vector"]
        v = np.array(vec, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"Vector for row {item['id']} must be one-dimensional")
        if v.shape != q.shape:
            raise ValueError(f"Dimension mismatch: query {q.shape} vs row {v.shape}")

        v = _l2_normalize(v)

        row = {
            "row_id": item["id"],
            "distance": _euclidean(q, v),
            "vector": vec
        }

        if item["payload"]:
            row["metadata"] = item["payload"]

        rows.append(row)

    rows.sort(key=lambda x: x["distance"])
    return rows[:k]


def normalize_vector(v: List[float]) -> List[float]:
    """Приводит вектор к единичной норме L2."""
    arr = np.array(v, dtype=float)
    normalized = _l2_normalize(arr)
    return normalized.tolist()