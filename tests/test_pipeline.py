"""Tests cover the decisions that are easy to get wrong and expensive to
debug in production: chunk boundaries, citation integrity, fusion order
and the offline embedder's determinism."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmind.config import Settings
from docmind.index.bm25 import BM25, tokenize
from docmind.ingest.chunker import chunk_document, chunk_section
from docmind.index.embedder import LsaEmbedder
from docmind.index.pipeline import build_index, load_index
from docmind.index.store import Store
from docmind.ingest.chunker import chunk_document
from docmind.ingest.loader import load_document
from docmind.retrieve.hybrid import rrf_fuse
from docmind.service import DocMind
from docmind.types import Chunk, Hit


def test_chunker_respects_size_and_overlap():
    text = ("The retention period is thirty days. " * 40).strip()
    chunks = chunk_section(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) < 260 for c in chunks)
    # consecutive chunks must share content, otherwise a sentence cut at a
    # window edge becomes unanswerable
    assert chunks[0].split()[-3:] != chunks[1].split()[:3]
    assert set(chunks[0].split()[-4:]) & set(chunks[1].split())


def test_chunks_never_cross_sections(tmp_path):
    (tmp_path / "doc.md").write_text(
        "# Policy\n\n## Refunds\n\nRefunds are available for 7 days.\n\n"
        "## Retention\n\nData is deleted after 30 days.\n",
        encoding="utf-8",
    )
    doc = load_document(tmp_path / "doc.md")
    chunks = chunk_document(doc, size=900, overlap=100)
    refund = [c for c in chunks if "Refunds" in c.heading]
    retention = [c for c in chunks if "Retention" in c.heading]
    assert refund and retention
    assert "30 days" not in refund[0].text
    assert "7 days" not in retention[0].text


def test_bm25_prefers_the_matching_document():
    corpus = [
        tokenize("The refund window is seven days for monthly plans."),
        tokenize("Uptime credits are applied to the next invoice."),
        tokenize("Backups are taken every six hours and kept 35 days."),
    ]
    scores = BM25().fit(corpus).scores(tokenize("how long are backups kept"))
    assert int(np.argmax(scores)) == 2


def test_lsa_embedding_is_deterministic_and_normalised():
    texts = ["data retention is thirty days", "backups every six hours", "refund window"]
    a = LsaEmbedder(dim=3).fit(texts)
    b = LsaEmbedder(dim=3).fit(texts)
    va = a.encode(["retention"])
    vb = b.encode(["retention"])
    assert np.allclose(va, vb)
    assert np.isclose(np.linalg.norm(va[0]), 1.0, atol=1e-5)


def test_rrf_prefers_agreement_between_retrievers():
    def hit(name, score):
        return Hit(chunk=Chunk(name, name, name, "h", "text", 0), score=score, source=name)

    a = [hit("x", 9.0), hit("y", 8.0)]
    b = [hit("y", 0.9), hit("z", 0.8)]
    fused = rrf_fuse([a, b], k=60)
    assert fused[0].chunk.chunk_id == "y"


@pytest.fixture()
def service(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "refunds.md").write_text(
        "# Refunds\n\n## Window\n\nRefunds are available within 7 days of the charge.\n",
        encoding="utf-8")
    (corpus / "backups.md").write_text(
        "# Backups\n\n## Schedule\n\nBackups run every six hours and are kept for 35 days.\n",
        encoding="utf-8")

    settings = Settings(
        root=tmp_path, corpus_dir=corpus,
        index_dir=tmp_path / "index", db_path=tmp_path / "index" / "test.db",
    )
    store = Store(settings.db_path)
    build_index(corpus, settings, store)
    return DocMind(load_index(settings, store), settings)


def test_end_to_end_search_and_answer(service):
    hits = service.search("how long are backups retained", mode="hybrid_rerank", top_k=3)
    assert hits[0].chunk.doc_id == "backups"

    answer = service.answer("how long are backups retained")
    assert answer.citations
    assert answer.mode == "extractive"
    assert "35 days" in answer.text


def test_answer_without_match_says_so(service):
    answer = service.answer("what is the airspeed velocity of an unladen swallow")
    assert answer.text
    assert answer.citations == [] or "No passage" in answer.text


def test_unknown_mode_is_rejected(service):
    with pytest.raises(ValueError):
        service.search("anything", mode="magic")
