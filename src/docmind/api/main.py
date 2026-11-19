"""FastAPI surface. Serves the JSON API and the bundled single-page UI."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..service import MODES, DocMind

app = FastAPI(title="DocMind RAG", version="1.0.0",
              description="Hybrid retrieval over your own documents, with citations.")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_state = {"service": None, "error": None}


def get_service() -> DocMind:
    if _state["service"] is None:
        try:
            _state["service"] = DocMind.load(settings)
        except Exception as exc:  # surfaced to the caller instead of a 500 traceback
            _state["error"] = str(exc)
            raise HTTPException(status_code=503, detail=str(exc))
    return _state["service"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = "hybrid_rerank"
    top_k: int = Field(8, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = "hybrid_rerank"


@app.get("/api/health")
def health():
    try:
        service = get_service()
    except HTTPException as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": exc.detail})
    return {
        "status": "ok",
        "chunks": service.index.size,
        "documents": len(service.documents),
        "embedder": service.index.embedder.name,
        "answer_mode": "llm" if settings.openai_api_key else "extractive",
        "modes": list(MODES),
    }


@app.get("/api/documents")
def documents():
    return {"documents": get_service().documents}


@app.post("/api/search")
def search(req: SearchRequest):
    started = time.time()
    hits = get_service().search(req.query, mode=req.mode, top_k=req.top_k)
    return {
        "query": req.query,
        "mode": req.mode,
        "latency_ms": round((time.time() - started) * 1000, 1),
        "hits": [h.to_dict() for h in hits],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    result = get_service().answer(req.question, mode=req.mode)
    return {
        "question": req.question,
        "answer": result.text,
        "mode": result.mode,
        "latency_ms": round(result.latency_ms, 1),
        "usage": result.usage,
        "citations": [
            {"index": c.index, "doc_id": c.doc_id, "doc_title": c.doc_title,
             "heading": c.heading, "snippet": c.snippet, "score": round(c.score, 4)}
            for c in result.citations
        ],
    }


WEB = Path(__file__).resolve().parents[3] / "web" / "index.html"


@app.get("/")
def index():
    if not WEB.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(WEB)
