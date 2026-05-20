from app.knn import knn_search
from app.storage import Storage


def test_knn_basic(tmp_path):
    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))
    storage.insert_vectors([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.2]
    ])

    result = knn_search(storage, [0.1, 0.1], k=2)

    assert len(result) == 2
    assert result[0]["vector"] == [0.0, 0.0]
    assert result[1]["vector"] == [0.2, 0.2]


def test_knn_dimension_mismatch(tmp_path):
    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))
    storage.insert_vectors([[0.0, 0.0]])

    try:
        knn_search(storage, [0.1, 0.1, 0.1], k=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)


def test_knn_empty_dataset(tmp_path):
    db_path = tmp_path / "storage.db"
    storage = Storage(str(db_path))

    result = knn_search(storage, [0.1, 0.2], k=3)
    assert result == []