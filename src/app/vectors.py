from typing import List

from .storage import Storage


def set_subjective_vector(storage: Storage, vector: List[float]):
    if not vector:
        raise ValueError("Vector cannot be empty")
    storage.set_subjective_vector([float(x) for x in vector])


def get_subjective_vector(storage: Storage):
    current = storage.get_subjective_vector()
    history = storage.get_subjective_history(limit=5)
    return {
        "current": current,
        "previous_versions": history
    }