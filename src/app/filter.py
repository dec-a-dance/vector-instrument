from typing import List
import numpy as np


def filter_by_subjective_dynamics(
    knn_results: List[dict],
    subjective_vector: List[float],
    params: dict
) -> List[dict]:
    if not subjective_vector:
        return knn_results

    weight = float(params.get("weight", 0.5))
    threshold = params.get("threshold", None)

    if weight < 0.0 or weight > 1.0:
        raise ValueError("weight must be between 0 and 1")

    s = np.array(subjective_vector, dtype=float)
    if s.ndim != 1:
        raise ValueError("subjective_vector must be one-dimensional")

    filtered = []
    for item in knn_results:
        v = np.array(item["vector"], dtype=float)
        if v.shape != s.shape:
            continue

        similarity = 1.0 / (1.0 + float(np.linalg.norm(v - s)))
        distance_score = 1.0 / (1.0 + float(item["distance"]))
        score = (1.0 - weight) * distance_score + weight * similarity

        enriched = dict(item)
        enriched["score"] = score

        if threshold is None or score >= float(threshold):
            filtered.append(enriched)

    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered