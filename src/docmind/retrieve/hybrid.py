"""Hybrid retrieval.

Lexical (BM25) and dense (embedding) retrieval fail in different places:
BM25 misses paraphrases, dense misses exact identifiers such as error
codes or plan names. We run both and fuse with Reciprocal Rank Fusion,
which needs no score calibration between the two — that matters because
BM25 scores and cosine similarities live on completely different scales.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..index.bm25 import tokenize
from ..index.pipeline import Index
from ..types import Hit


def bm25_search(index: Index, query: str, depth: int) -> List[Hit]:
    scores = index.bm25.scores(tokenize(query))
    order = np.argsort(scores)[::-1][:depth]
    return [
        Hit(chunk=index.chunks[i], score=float(scores[i]), source="bm25")
        for i in order if scores[i] > 0
    ]


def dense_search(index: Index, query: str, depth: int) -> List[Hit]:
    if index.vectors.size == 0:
        return []
    q = index.embedder.encode([query], is_query=True)[0]
    sims = index.vectors @ q
    order = np.argsort(sims)[::-1][:depth]
    return [
        Hit(chunk=index.chunks[i], score=float(sims[i]), source="dense")
        for i in order
    ]


def rrf_fuse(result_sets: Sequence[Sequence[Hit]], k: int = 60,
             weights: Sequence[float] = None) -> List[Hit]:
    weights = list(weights or [1.0] * len(result_sets))
    scores: dict = {}
    holders: dict = {}
    parts: dict = {}

    for weight, hits in zip(weights, result_sets):
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk.chunk_id
            holders.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            parts.setdefault(key, {})[hit.source] = hit.score

    fused = [
        Hit(chunk=holders[key].chunk, score=score, source="fused", components=parts[key])
        for key, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused


def hybrid_search(index: Index, query: str, depth: int = 40, k: int = 60) -> List[Hit]:
    return rrf_fuse([bm25_search(index, query, depth), dense_search(index, query, depth)], k=k)
