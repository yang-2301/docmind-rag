"""Chunking.

Two decisions worth calling out, because most naive RAG demos get them
wrong:

1. Chunk inside a section, never across one. A chunk that spans two
   policy sections produces an answer that cites the right document and
   the wrong rule.
2. Split on sentence boundaries with a target size in characters,
   carrying an overlap so a sentence cut at the window edge still
   appears whole in the neighbouring chunk.
"""
from __future__ import annotations

import re
from typing import List

from ..types import Chunk, Document

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WS = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    text = _WS.sub(" ", text).strip()
    if not text:
        return []
    parts = _SENTENCE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(text: str, size: int, overlap: int) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    length = 0

    for sentence in sentences:
        if buf and length + len(sentence) + 1 > size:
            chunks.append(" ".join(buf))
            # carry the tail of the previous window forward as overlap
            carry: List[str] = []
            carried = 0
            for prev in reversed(buf):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            buf, length = carry, carried
        buf.append(sentence)
        length += len(sentence) + 1

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for section in doc.sections:
        for i, body in enumerate(chunk_section(section.text, size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id="%s::%d::%d" % (doc.doc_id, section.order, i),
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    heading=section.heading,
                    text=body,
                    position=section.order,
                )
            )
    return chunks
