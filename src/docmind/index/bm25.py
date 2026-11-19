"""A small, dependency-free BM25 (Okapi) implementation.

Why not a library: the lexical half of a hybrid retriever is ~60 lines
of maths, and keeping it here means the repo installs nothing beyond
numpy/fastapi and behaves identically everywhere.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "as", "at", "by", "from", "if", "we", "you", "i", "our", "your",
    "can", "will", "may", "not", "no", "do", "does", "how", "what", "when",
    "which", "who", "there", "their", "they", "has", "have", "had", "but",
}


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN.findall(text.lower())
    out = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        # crude but effective: fold simple plurals so "refund" matches "refunds"
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avg_len = 0.0

    def fit(self, docs_tokens: Sequence[Sequence[str]]) -> "BM25":
        self.doc_freqs = [Counter(tokens) for tokens in docs_tokens]
        self.doc_len = [len(tokens) for tokens in docs_tokens]
        n = len(docs_tokens)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0

        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())

        # BM25 idf with the +0.5 smoothing, floored at 0 so that very
        # common terms cannot contribute negatively.
        self.idf = {
            term: max(0.0, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in df.items()
        }
        return self

    def scores(self, query_tokens: Iterable[str]) -> List[float]:
        q = list(query_tokens)
        out = [0.0] * len(self.doc_freqs)
        if not self.doc_freqs or not q:
            return out

        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i] or 1
            score = 0.0
            for term in q:
                f = freqs.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                score += idf * (f * (self.k1 + 1)) / denom
            out[i] = score
        return out
