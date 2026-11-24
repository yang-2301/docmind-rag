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
from docmind.service import DocMind, MODES
from docmind.generate.answer import answer as generate_answer
from docmind.retrieve.grounding import check_grounding
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


# --- grounding gate -----------------------------------------------------

def test_grounding_gate_refuses_an_unrelated_question(service):
    result = service.answer("what is the airspeed velocity of an unladen swallow")
    assert result.grounded is False
    assert result.citations == []
    assert "No passage" in result.text
    assert result.grounding["coverage"] == 0.0
    assert result.grounding["matched_terms"] == []


def test_grounding_gate_exposes_its_reasoning(service):
    result = service.answer("how long are backups retained")
    assert result.grounded is True
    g = result.grounding
    assert g["coverage"] >= g["threshold"]
    assert "backup" in g["matched_terms"]
    assert g["query_terms"] and g["passages_checked"] > 0


def test_grounding_uses_union_of_passages_not_a_single_hit():
    # Neither chunk alone covers the question, but together they do.
    q = "refund window and retention period"
    a = Hit(Chunk("a", "a", "A", "h", "The refund window is seven days.", 0), 1.0)
    b = Hit(Chunk("b", "b", "B", "h", "The retention period is thirty days.", 0), 1.0)
    single = check_grounding(q, [a], 0.3)
    both = check_grounding(q, [a, b], 0.3)
    assert single.coverage < both.coverage
    assert both.coverage == 1.0


def test_citation_number_is_the_retrieved_rank(tmp_path):
    """[2] must point at passage 02 even when passage 01 is unusable."""
    settings = Settings(root=tmp_path, index_dir=tmp_path / "index",
                        db_path=tmp_path / "index" / "x.db")
    first = Hit(Chunk("first", "d1", "Doc One", "Overview",
                      "Nothing relevant appears in this passage.", 0), 10.0)
    second = Hit(Chunk("second", "d2", "Doc Two", "Schedule",
                       "Backups are kept for 35 days.", 0), 9.0)
    result = generate_answer("how long are backups kept", [first, second], settings)
    assert result.citations
    assert result.citations[0].index == 2
    assert "[2]" in result.text
    assert "35 days" in result.text


def test_weak_second_passage_is_not_cited(tmp_path):
    """A second hit sharing one generic word must not earn a citation."""
    settings = Settings(root=tmp_path, index_dir=tmp_path / "index",
                        db_path=tmp_path / "index" / "x.db")
    good = Hit(Chunk("good", "refunds", "Refunds", "Window",
                     "Refunds take five business days to reach the original payment method.", 0), 1.0)
    weak = Hit(Chunk("weak", "discounts", "Discounts", "Overview",
                     "Discount statements arrive annually in advance.", 0), 0.4)
    result = generate_answer("how long does a refund take to arrive", [good, weak], settings)
    assert [c.doc_id for c in result.citations] == ["refunds"]
    assert "[2]" not in result.text


# --- new service surface ------------------------------------------------

def test_documents_reports_chunk_counts_and_headings(service):
    docs = {d["doc_id"]: d for d in service.documents}
    assert set(docs) == {"refunds", "backups"}
    assert all(d["chunks"] >= 1 and d["headings"] for d in docs.values())


def test_document_detail_is_verifiable_and_unknown_ids_return_none(service):
    detail = service.document("backups")
    assert detail["title"] == "Backups"
    assert any("35 days" in p["text"] for s in detail["sections"] for p in s["passages"])
    assert service.document("does-not-exist") is None


def test_compare_runs_every_mode_on_the_same_question(service):
    comparison = service.compare("how long are backups retained", top_k=2)
    assert [row["mode"] for row in comparison] == list(MODES)
    assert all(0 < len(row["hits"]) <= 2 for row in comparison)
    assert all(row["latency_ms"] >= 0 for row in comparison)


# --- API ----------------------------------------------------------------

def test_api_endpoints(service, monkeypatch):
    import docmind.api.main as api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "_state", {"service": service, "error": None})
    client = TestClient(api.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert "grounding_min_coverage" in health.json()

    docs = client.get("/api/documents").json()["documents"]
    assert any(d["doc_id"] == "backups" for d in docs)
    assert client.get("/api/documents/backups").json()["chunks"] >= 1
    assert client.get("/api/documents/missing").status_code == 404

    asked = client.post("/api/ask", json={
        "question": "how long are backups retained", "mode": "hybrid_rerank", "top_k": 6,
    }).json()
    assert asked["grounded"] is True
    assert asked["grounding"]["coverage"] > 0
    assert all(c["index"] <= 6 for c in asked["citations"])

    refused = client.post("/api/ask", json={
        "question": "how do I bake sourdough bread", "top_k": 6,
    }).json()
    assert refused["grounded"] is False
    assert refused["citations"] == []

    comparison = client.post("/api/compare", json={"query": "how long are backups retained"}).json()
    assert len(comparison["comparison"]) == len(MODES)
