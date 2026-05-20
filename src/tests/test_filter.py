import pytest

from app.filter import filter_by_subjective_dynamics


def test_filter_without_subjective_vector():
    knn_results = [
        {"row_id": 1, "distance": 0.1, "vector": [0.1, 0.2]}
    ]

    result = filter_by_subjective_dynamics(knn_results, [], {"weight": 0.5})
    assert result == knn_results


def test_filter_with_subjective_vector():
    knn_results = [
        {"row_id": 1, "distance": 0.1, "vector": [0.1, 0.2]},
        {"row_id": 2, "distance": 0.2, "vector": [0.9, 0.9]}
    ]
    subjective = [0.1, 0.2]

    result = filter_by_subjective_dynamics(
        knn_results,
        subjective,
        {"weight": 0.7}
    )

    assert len(result) == 2
    assert result[0]["row_id"] == 1
    assert "score" in result[0]


def test_filter_invalid_weight():
    with pytest.raises(ValueError, match="weight must be between 0 and 1"):
        filter_by_subjective_dynamics([], [0.1, 0.2], {"weight": 1.5})