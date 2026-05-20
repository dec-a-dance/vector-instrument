from typing import List, Optional
import numpy as np


def _to_array(value) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Vector must be one-dimensional")
    return arr


def _get_change_direction(
    subjective_vector: List[float],
    subjective_history: List[dict] = None
) -> np.ndarray:
    current = _to_array(subjective_vector)

    if not subjective_history:
        return np.zeros_like(current)

    history_vectors = []
    for item in subjective_history:
        try:
            hist_vec = _to_array(item)
            if hist_vec.shape == current.shape:
                history_vectors.append(hist_vec)
        except Exception:
            continue

    if not history_vectors:
        return np.zeros_like(current)

    deltas = []
    prev = history_vectors[-1]
    deltas.append(current - prev)

    for i in range(len(history_vectors) - 1, 0, -1):
        deltas.append(history_vectors[i] - history_vectors[i - 1])

    return np.mean(np.array(deltas), axis=0)


def filter_by_subjective_dynamics(
    knn_results: List[dict],
    subjective_vector: List[float],
    subjective_history: List[dict],
    params: dict
) -> List[dict]:
    if not subjective_vector:
        return knn_results

    threshold = params.get("threshold", 0.5)
    direction_weight = float(params.get("direction_weight", 1.0))

    if direction_weight < 0.0 or direction_weight > 1.0:
        raise ValueError("direction_weight must be between 0 and 1")

    s = _to_array(subjective_vector)
    direction = _get_change_direction(subjective_vector, subjective_history)

    filtered = []
    for item in knn_results:
        v = np.array(item["vector"], dtype=float)
        if v.ndim != 1 or v.shape != s.shape:
            continue

        distance_to_subjective = float(np.linalg.norm(v - s))

        if np.all(direction == 0):
            direction_score = 0.0
        else:
            signed_alignment = np.sign(v - s) * np.sign(direction)
            direction_score = float(np.mean(signed_alignment))

        score = (
            (1.0 - direction_weight) * (1.0 / (1.0 + distance_to_subjective))
            + direction_weight * (0.5 * (direction_score + 1.0))
        )

        enriched = dict(item)
        enriched["score"] = score
        enriched["distance_to_subjective"] = distance_to_subjective
        enriched["direction_score"] = direction_score

        if threshold is None or score >= float(threshold):
            filtered.append(enriched)

    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered