"""Query service: one entry point shared by the API, the CLI and the
evaluation harness, so all three measure exactly the same pipeline."""
from __future__ import annotations

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
        hits = self.search(question, mode=mode, top_k=top_k)
        return generate_answer(question, hits[: self.settings.top_k], self.settings)

    @property
    def documents(self) -> List[dict]:
        seen = {}
        for chunk in self.index.chunks:
            seen.setdefault(chunk.doc_id, chunk.doc_title)
        return [{"doc_id": k, "title": v} for k, v in seen.items()]
