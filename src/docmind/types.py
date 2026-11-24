"""Core data structures shared across ingest, retrieval and generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Section:
    """A document section, produced by splitting on markdown headings.

    Sections are the unit of citation: an answer points at a document
    title plus a heading, never at an opaque chunk id.
    """
    heading: str
    text: str
    order: int


@dataclass
class Document:
    doc_id: str
    title: str
    path: str
    sections: List[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    position: int          # order within the document, for stable citations


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = ""       # bm25 | dense | fused | rerank
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "heading": self.chunk.heading,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "source": self.source,
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


@dataclass
class Citation:
    index: int
    doc_id: str
    doc_title: str
    heading: str
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    citations: List[Citation]
    mode: str                        # "llm" | "extractive"
    latency_ms: float = 0.0
    usage: Optional[Dict[str, int]] = None
    grounded: bool = False           # did the grounding gate allow generation?
    grounding: Optional[Dict] = None  # coverage / matched terms / threshold

