"""Embedding backends behind one interface.

The default backend is TF-IDF + truncated SVD (latent semantic analysis).
It is not as good as a trained encoder, but it is:
  * deterministic,
  * offline (no model download, no GPU),
  * good enough that the retrieval evaluation in this repo runs on any
    machine with numpy.

Swap `DOCMIND_EMBEDDER=openai` or `minilm` to use a real encoder; the
rest of the pipeline is unchanged because everything goes through
`fit()` / `encode()`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .bm25 import tokenize


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BaseEmbedder:
    dim: int = 0
    name: str = "base"

    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":
        raise NotImplementedError

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> bool:
        return False


class LsaEmbedder(BaseEmbedder):
    name = "lsa"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab: List[str] = []
        self.index = {}
        self.idf: np.ndarray = np.zeros(0)
        self.components: np.ndarray = np.zeros((0, 0))

    # ---------- fitting ----------
    def _counts(self, texts: Sequence[str]) -> List[dict]:
        rows = []
        for text in texts:
            counts: dict = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            rows.append(counts)
        return rows

    def fit(self, texts: Sequence[str]) -> "LsaEmbedder":
        rows = self._counts(texts)
        n = len(rows)
        df: dict = {}
        for row in rows:
            for term in row:
                df[term] = df.get(term, 0) + 1

        # keep the most informative terms; drop one-off hapax noise only
        # when the vocabulary would otherwise explode
        terms = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(terms) > 20000:
            terms = terms[:20000]
        self.vocab = [t for t, _ in terms]
        self.index = {t: i for i, t in enumerate(self.vocab)}

        self.idf = np.array(
            [np.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab], dtype=np.float32
        )

        mat = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, row in enumerate(rows):
            for term, count in row.items():
                j = self.index.get(term)
                if j is not None:
                    mat[i, j] = (1.0 + np.log(count)) * self.idf[j]
        mat = _l2_normalize(mat)

        k = int(min(self.dim, max(1, min(mat.shape) - 1)))
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        self.components = np.ascontiguousarray(vt[:k], dtype=np.float32)
        self.dim = k
        return self

    # ---------- encoding ----------
    def _vectorize(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for term, count in counts.items():
                j = self.index.get(term)
                if j is not None:
                    mat[i, j] = (1.0 + np.log(count)) * self.idf[j]
        return _l2_normalize(mat)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not len(self.vocab):
            return np.zeros((len(texts), 0), dtype=np.float32)
        return _l2_normalize(self._vectorize(texts) @ self.components.T)

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "lsa.npz",
            idf=self.idf,
            components=self.components,
            vocab=np.array(self.vocab, dtype=object),
        )

    def load(self, path: Path) -> bool:
        file = path / "lsa.npz"
        if not file.exists():
            return False
        data = np.load(file, allow_pickle=True)
        self.vocab = list(data["vocab"])
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.idf = data["idf"]
        self.components = data["components"]
        self.dim = int(self.components.shape[0])
        return True


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        from openai import OpenAI  # imported lazily so the dep stays optional

        self.client = OpenAI(api_key=api_key)
        self.dim = 1536

    def fit(self, texts: Sequence[str]) -> "OpenAIEmbedder":
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        vectors: List[List[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(model=self.model, input=list(texts[i:i + batch]))
            vectors.extend([item.embedding for item in resp.data])
        arr = np.array(vectors, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return _l2_normalize(arr)


class MiniLMEmbedder(BaseEmbedder):
    name = "minilm"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts: Sequence[str]) -> "MiniLMEmbedder":
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        arr = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)


def build_embedder(kind: str, dim: int = 128, model: str = "text-embedding-3-small",
                   api_key: Optional[str] = None) -> BaseEmbedder:
    kind = (kind or "lsa").lower()
    if kind == "openai":
        if not api_key:
            raise RuntimeError("DOCMIND_EMBEDDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbedder(model=model, api_key=api_key)
    if kind == "minilm":
        return MiniLMEmbedder()
    return LsaEmbedder(dim=dim)
