import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any


class Storage:
    def __init__(self, db_path: str = "vector_storage.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_json TEXT NOT NULL,
                    payload_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subjective_vectors (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def clear_vectors(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM vectors")
            conn.commit()

    def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None
    ):
        if payloads is not None and len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")

        with self._connect() as conn:
            if payloads is None:
                data = [(json.dumps(v), None) for v in vectors]
            else:
                data = [
                    (json.dumps(v), json.dumps(p, ensure_ascii=False))
                    for v, p in zip(vectors, payloads)
                ]

            conn.executemany(
                "INSERT INTO vectors (vector_json, payload_json) VALUES (?, ?)",
                data
            )
            conn.commit()

    def fetch_all_vectors(self) -> List[List[float]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, vector_json FROM vectors ORDER BY id").fetchall()
            return [json.loads(row["vector_json"]) for row in rows]

    def fetch_vectors_with_payloads(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT id, vector_json, payload_json FROM vectors"
        params: List[Any] = []
        where_clauses: List[str] = []

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"json_extract(payload_json, '$.{key}') = ?")
                params.append(value)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY id"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
                result.append({
                    "id": int(row["id"]),
                    "vector": json.loads(row["vector_json"]),
                    "payload": payload
                })
            return result

    def count_vectors(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM vectors").fetchone()
            return int(row["cnt"])

    def set_metadata(self, key: str, value: str):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, value))
            conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_subjective_vector(self, vector: List[float]):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO subjective_vectors (vector_json) VALUES (?)",
                (json.dumps(vector),)
            )
            conn.commit()

    def get_subjective_vector(self) -> Optional[List[float]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT vector_json
                FROM subjective_vectors
                ORDER BY version DESC
                LIMIT 1
            """).fetchone()
            return json.loads(row["vector_json"]) if row else None

    def get_subjective_history(self, limit: int = 10) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT version, vector_json, created_at
                FROM subjective_vectors
                ORDER BY version DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [
                {
                    "version": int(row["version"]),
                    "vector": json.loads(row["vector_json"]),
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

    def reset_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM vectors")
            conn.execute("DELETE FROM metadata")
            conn.execute("DELETE FROM subjective_vectors")
            conn.commit()

    def stats(self) -> dict:
        return {
            "vector_count": self.count_vectors(),
            "dimensions": json.loads(self.get_metadata("dimensions") or "null"),
            "source_type": self.get_metadata("source_type"),
            "has_subjective_vector": self.get_subjective_vector() is not None
        }