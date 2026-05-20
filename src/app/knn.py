from typing import List
import numpy as np

from .storage import Storage


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def knn_search(storage: Storage, query_vector: List[float], k: int = 5) -> List[dict]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dataset = storage.fetch_all_vectors()
    if not dataset:
        return []

    q = np.array(query_vector, dtype=float)
    if q.ndim != 1:
        raise ValueError("Query vector must be one-dimensional")

    rows = []
    for idx, vec in enumerate(dataset, start=1):
        v = np.array(vec, dtype=float)
        if v.shape != q.shape:
            raise ValueError(f"Dimension mismatch: query {q.shape} vs row {v.shape}")
        rows.append({
            "row_id": idx,
            "distance": _euclidean(q, v),
            "vector": vec
        })

    rows.sort(key=lambda x: x["distance"])
    return rows[:k]