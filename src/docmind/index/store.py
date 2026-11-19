"""Persistence: SQLite for chunks + vectors, numpy files for the embedder.

SQLite was chosen over a vector service on purpose — the whole point of
this repo is that `git clone && make dev` works on a laptop with no
Docker, no Postgres, no model server. The vector column is a plain
float32 blob; swapping this module for a pgvector implementation is a
~50 line change and does not touch retrieval or generation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ..types import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    doc_title  TEXT NOT NULL,
    heading    TEXT NOT NULL,
    position   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.executescript(SCHEMA)
        return conn

    def reset(self) -> None:
        conn = self._connect()
        with conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM meta")
        conn.close()

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        rows: List[Tuple] = []
        for chunk, vec in zip(chunks, vectors):
            rows.append((
                chunk.chunk_id, chunk.doc_id, chunk.doc_title, chunk.heading,
                chunk.position, chunk.text,
                np.asarray(vec, dtype=np.float32).tobytes(),
            ))
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, doc_title, heading, position, text, embedding) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        conn.close()

    def load(self) -> Tuple[List[Chunk], np.ndarray]:
        conn = self._connect()
        cur = conn.execute(
            "SELECT chunk_id, doc_id, doc_title, heading, position, text, embedding "
            "FROM chunks ORDER BY rowid"
        )
        chunks: List[Chunk] = []
        vectors: List[np.ndarray] = []
        for row in cur.fetchall():
            chunks.append(Chunk(
                chunk_id=row[0], doc_id=row[1], doc_title=row[2],
                heading=row[3], position=row[4], text=row[5],
            ))
            vectors.append(np.frombuffer(row[6], dtype=np.float32))
        conn.close()
        if not vectors:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        return chunks, np.vstack(vectors)

    def count(self) -> int:
        conn = self._connect()
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        return n

    def set_meta(self, key: str, value: str) -> None:
        conn = self._connect()
        with conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.close()

    def get_meta(self, key: str, default: str = "") -> str:
        conn = self._connect()
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        conn.close()
        return row[0] if row else default
