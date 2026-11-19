"""Ingestion pipeline: directory of documents -> persisted, queryable index."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from ..config import Settings
from ..types import Chunk
from ..ingest.chunker import chunk_document
from ..ingest.loader import load_corpus
from .bm25 import BM25, tokenize
from .embedder import BaseEmbedder, build_embedder
from .store import Store


@dataclass
class Index:
    chunks: List[Chunk]
    vectors: np.ndarray
    bm25: BM25
    embedder: BaseEmbedder

    @property
    def size(self) -> int:
        return len(self.chunks)


def build_index(corpus_dir: Path, settings: Settings, store: Store,
                embedder: BaseEmbedder = None) -> Index:
    docs = load_corpus(corpus_dir)
    if not docs:
        raise RuntimeError("no documents found in %s" % corpus_dir)

    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))

    embedder = embedder or build_embedder(
        settings.embedder, dim=settings.lsa_dim,
        model=settings.embed_model, api_key=settings.openai_api_key,
    )
    embedder.fit([c.text for c in chunks])
    vectors = embedder.encode([c.text for c in chunks])

    store.reset()
    store.add(chunks, vectors)
    embedder.save(settings.index_dir)
    store.set_meta("docs", json.dumps([{"doc_id": d.doc_id, "title": d.title} for d in docs]))
    store.set_meta("embedder", embedder.name)

    bm25 = BM25().fit([tokenize(c.text) for c in chunks])
    return Index(chunks=chunks, vectors=vectors, bm25=bm25, embedder=embedder)


def load_index(settings: Settings, store: Store) -> Index:
    chunks, vectors = store.load()
    if not chunks:
        raise RuntimeError("index is empty — run `make ingest` first")

    embedder = build_embedder(
        store.get_meta("embedder", settings.embedder), dim=settings.lsa_dim,
        model=settings.embed_model, api_key=settings.openai_api_key,
    )
    if not embedder.load(settings.index_dir) and embedder.name == "lsa":
        raise RuntimeError("embedder artefacts missing — run `make ingest` first")

    bm25 = BM25().fit([tokenize(c.text) for c in chunks])
    return Index(chunks=chunks, vectors=vectors, bm25=bm25, embedder=embedder)
