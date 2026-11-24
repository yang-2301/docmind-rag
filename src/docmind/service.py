"""Query service: one entry point shared by the API, the CLI and the
evaluation harness, so all three measure exactly the same pipeline."""
from __future__ import annotations

import time
from typing import List

from .config import Settings, settings as default_settings
from .generate.answer import answer as generate_answer
from .index.pipeline import Index, load_index
from .index.store import Store
from .retrieve.hybrid import bm25_search, dense_search, hybrid_search
from .retrieve.rerank import rerank
from .types import Answer, Hit

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class DocMind:
    def __init__(self, index: Index, settings: Settings):
        self.index = index
        self.settings = settings

    @classmethod
    def load(cls, settings: Settings = None) -> "DocMind":
        settings = settings or default_settings
        store = Store(settings.db_path)
        return cls(load_index(settings, store), settings)

    # ---------- retrieval ----------
    def search(self, query: str, mode: str = "hybrid_rerank",
               top_k: int = None) -> List[Hit]:
        if mode not in MODES:
            raise ValueError("unknown mode %r, expected one of %s" % (mode, MODES))

        top_k = top_k or self.settings.top_k
        depth = self.settings.candidate_k

        if mode == "bm25":
            return bm25_search(self.index, query, top_k)
        if mode == "dense":
            return dense_search(self.index, query, top_k)

        fused = hybrid_search(self.index, query, depth=depth, k=self.settings.rrf_k)
        if mode == "hybrid":
            return fused[:top_k]
        return rerank(self.index, query, fused[:depth], top_k)

    # ---------- end to end ----------
    def answer(self, question: str, top_k: int = None,
               mode: str = "hybrid_rerank") -> Answer:
        top_k = top_k or self.settings.top_k
        hits = self.search(question, mode=mode, top_k=top_k)
        # Truncate to exactly the depth the caller asked for: citation
        # numbers are ranks in this list, and the UI displays the same list.
        return generate_answer(question, hits[:top_k], self.settings)

    @property
    def documents(self) -> List[dict]:
        seen = {}
        for chunk in self.index.chunks:
            entry = seen.setdefault(chunk.doc_id, {
                "doc_id": chunk.doc_id, "title": chunk.doc_title,
                "chunks": 0, "headings": [],
            })
            entry["chunks"] += 1
            if chunk.heading not in entry["headings"]:
                entry["headings"].append(chunk.heading)
        return list(seen.values())

    def document(self, doc_id: str) -> dict:
        """Full text of one indexed document, grouped by section.

        Lets a reader verify a citation against the source instead of
        trusting the snippet. Returns None when the id is unknown, so the
        API can answer 404 rather than silently returning an empty page.
        """
        chunks = [c for c in self.index.chunks if c.doc_id == doc_id]
        if not chunks:
            return None
        sections = {}
        for c in chunks:
            sections.setdefault(c.heading, []).append({
                "chunk_id": c.chunk_id, "text": c.text, "position": c.position,
            })
        return {
            "doc_id": doc_id,
            "title": chunks[0].doc_title,
            "chunks": len(chunks),
            "sections": [
                {"heading": heading, "passages": sorted(parts, key=lambda p: p["position"])}
                for heading, parts in sections.items()
            ],
        }

    def compare(self, query: str, top_k: int = 3) -> List[dict]:
        """Run every retrieval mode on the same question.

        One query, four pipelines: this is the interactive version of the
        table in the README, and the fastest way to see where fusion or
        reranking actually changes the top hit.
        """
        out = []
        for mode in MODES:
            started = time.perf_counter()
            hits = self.search(query, mode=mode, top_k=top_k)
            latency = (time.perf_counter() - started) * 1000
            out.append({
                "mode": mode,
                "latency_ms": round(latency, 2),
                "hits": [h.to_dict() for h in hits],
            })
        return out
