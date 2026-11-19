"""Answer generation.

Two modes, chosen at runtime:

* `llm`        — an OpenAI-compatible chat model, instructed to answer
                 only from the supplied passages and to cite them as [n].
* `extractive` — no key, no network. Returns the highest-scoring
                 sentences verbatim, with citations. Weaker prose, but
                 it never invents a policy clause, which for a support
                 tool is the failure mode that actually matters.

Both modes return the same structure, so the UI never has to care which
one produced the answer.
"""
from __future__ import annotations

import re
import time
from typing import List, Sequence

from ..config import Settings
from ..types import Answer, Citation, Hit

SYSTEM_PROMPT = (
    "You answer questions about a company's internal documentation. "
    "Use ONLY the numbered passages provided. If the passages do not "
    "contain the answer, say so plainly and name what is missing — do "
    "not guess. Cite every factual sentence with the passage number in "
    "square brackets, like [2]. Keep answers under 120 words and prefer "
    "concrete numbers, deadlines and names over generalities."
)


def _snippet(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_citations(hits: Sequence[Hit]) -> List[Citation]:
    return [
        Citation(index=i, doc_id=h.chunk.doc_id, doc_title=h.chunk.doc_title,
                 heading=h.chunk.heading, snippet=_snippet(h.chunk.text), score=h.score)
        for i, h in enumerate(hits, start=1)
    ]


def _context_block(hits: Sequence[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            "[%d] %s — %s\n%s" % (i, hit.chunk.doc_title, hit.chunk.heading, hit.chunk.text)
        )
    return "\n\n".join(blocks)


def _extractive(question: str, hits: Sequence[Hit]) -> Answer:
    from ..index.bm25 import tokenize

    q_terms = set(tokenize(question))
    picked: List[str] = []
    used: List[Citation] = []

    # Only passages close to the best score are allowed to contribute.
    # Without this, a chunk that merely shares one word ("webhook") gets
    # cited next to the real answer and the citations stop meaning anything.
    cutoff = 0.45 * (hits[0].score if hits else 0.0)

    for hit in [h for h in hits if h.score >= cutoff][:2]:
        sentences = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
        scored = []
        for sentence in sentences:
            overlap = len(q_terms & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap / (len(q_terms) or 1), sentence))
        if not scored:
            continue
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = [s for ratio, s in scored[:2] if ratio >= 0.2]
        if not best:
            continue
        marker = len(used) + 1
        used.append(Citation(index=marker, doc_id=hit.chunk.doc_id,
                             doc_title=hit.chunk.doc_title, heading=hit.chunk.heading,
                             snippet=_snippet(hit.chunk.text), score=hit.score))
        picked.append(" ".join(best).strip() + " [%d]" % marker)
        if len(picked) >= 2:
            break

    body = "\n\n".join(picked) if picked else "No passage in the corpus covers this question."
    return Answer(text=body, citations=used, mode="extractive")


def answer(question: str, hits: Sequence[Hit], settings: Settings) -> Answer:
    started = time.time()

    from ..index.bm25 import tokenize

    q_terms = set(tokenize(question))
    overlap = any(q_terms & set(tokenize(h.chunk.text)) for h in list(hits)[:5])
    if not hits or not overlap:
        # Refuse before generation. A grounded system that answers anyway is
        # worse than one that says "I don't know".
        return Answer(text="No passage in the corpus covers this question.",
                      citations=[], mode="extractive",
                      latency_ms=(time.time() - started) * 1000)

    if not settings.openai_api_key:
        result = _extractive(question, hits)
        result.latency_ms = (time.time() - started) * 1000
        return result

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Passages:\n%s\n\nQuestion: %s"
                                        % (_context_block(hits), question)},
        ],
    )
    text = (completion.choices[0].message.content or "").strip()

    used_ids = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    citations = [c for c in build_citations(hits) if c.index in used_ids] or build_citations(hits[:3])

    usage = None
    if completion.usage:
        usage = {"prompt_tokens": completion.usage.prompt_tokens,
                 "completion_tokens": completion.usage.completion_tokens}

    return Answer(text=text, citations=citations, mode="llm",
                  latency_ms=(time.time() - started) * 1000, usage=usage)
