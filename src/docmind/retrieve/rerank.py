"""Reranking.

Fusion orders candidates by *rank consensus*, which is cheap but blind:
it never looks at the query and the passage together. A reranker scores
the pair directly, which is where most of the precision comes from.

Two implementations:

* `HeuristicReranker` — the default. Needs no model download. It scores
  idf-weighted term coverage, bigram agreement, term proximity (a query
  whose terms appear in one sentence beats one where they are scattered
  across a page) and heading agreement.
* `CrossEncoderReranker` — used automatically when sentence-transformers
  is installed. Same interface, better quality.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..index.bm25 import tokenize
from ..types import Hit


def _positions(tokens: Sequence[str], terms: Sequence[str]) -> Dict[str, List[int]]:
    wanted = set(terms)
    pos: Dict[str, List[int]] = {}
    for i, tok in enumerate(tokens):
        if tok in wanted:
            pos.setdefault(tok, []).append(i)
    return pos


class HeuristicReranker:
    name = "heuristic"

    def __init__(self, idf: Dict[str, float]):
        self.idf = idf
        self.default_idf = max(idf.values()) if idf else 1.0

    def _term_weight(self, term: str) -> float:
        return self.idf.get(term, self.default_idf * 0.5)

    def score(self, query: str, text: str, heading: str = "") -> Dict[str, float]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        d_tokens = tokenize(text)
        if not q_terms or not d_tokens:
            return {"coverage": 0.0, "bigram": 0.0, "proximity": 0.0, "heading": 0.0}

        total_weight = sum(self._term_weight(t) for t in q_terms) or 1.0
        d_set = set(d_tokens)

        # 1. idf-weighted coverage
        matched = [t for t in q_terms if t in d_set]
        coverage = sum(self._term_weight(t) for t in matched) / total_weight

        # 2. bigram agreement: adjacent query terms appearing adjacent in the text
        q_bigrams = list(zip(q_terms, q_terms[1:]))
        bigram = 0.0
        if q_bigrams:
            d_bigrams = set(zip(d_tokens, d_tokens[1:]))
            hits = sum(1 for bg in q_bigrams if bg in d_bigrams)
            bigram = hits / len(q_bigrams)

        # 3. proximity: tightest window covering any two distinct query terms
        pos = _positions(d_tokens, matched)
        proximity = 0.0
        if len(pos) >= 2:
            flat = sorted((p, t) for t, ps in pos.items() for p in ps)
            best = None
            for i in range(len(flat)):
                seen = {flat[i][1]}
                for j in range(i + 1, len(flat)):
                    seen.add(flat[j][1])
                    if len(seen) >= 2:
                        width = flat[j][0] - flat[i][0] + 1
                        best = width if best is None else min(best, width)
                        break
            if best:
                proximity = 1.0 / (1.0 + (best - 2) / 10.0)

        # 4. heading agreement
        h_terms = set(tokenize(heading))
        heading_score = (len(h_terms & set(q_terms)) / len(q_terms)) if q_terms else 0.0

        return {
            "coverage": round(coverage, 4),
            "bigram": round(bigram, 4),
            "proximity": round(proximity, 4),
            "heading": round(heading_score, 4),
        }

    def total(self, comp: Dict[str, float]) -> float:
        return (0.55 * comp["coverage"]
                + 0.20 * comp["bigram"]
                + 0.15 * comp["proximity"]
                + 0.10 * comp["heading"])


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: Sequence[Hit], top_k=None) -> List[Hit]:
        if not hits:
            return []
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs)
        out = [Hit(chunk=h.chunk, score=float(s), source="rerank", components=h.components)
               for h, s in zip(hits, scores)]
        out.sort(key=lambda h: h.score, reverse=True)
        return out[:top_k] if top_k else out


def build_reranker(index) -> object:
    try:
        return CrossEncoderReranker()
    except Exception:
        return HeuristicReranker(index.bm25.idf)


def rerank(index, query: str, hits: Sequence[Hit], top_k: int) -> List[Hit]:
    ranker = build_reranker(index)
    if isinstance(ranker, HeuristicReranker):
        scored = []
        for hit in hits:
            comp = ranker.score(query, hit.chunk.text, hit.chunk.heading)
            scored.append(Hit(chunk=hit.chunk, score=ranker.total(comp),
                              source="rerank", components=comp))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]
    return ranker.rerank(query, hits, top_k)
