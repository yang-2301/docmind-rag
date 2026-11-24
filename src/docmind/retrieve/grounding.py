"""Grounding gate: decide whether the retrieved passages actually support
answering, *before* any text is generated.

The naive version of this check is "do the question and any passage share at
least one word?" That is far too permissive on a policy corpus: a question
about penguins will match "explain", and a confident model will happily turn
that single word into an answer with a citation attached.

Instead we measure how much of the question is covered by the union of the
top passages. Union rather than a single passage matters because an answer is
often spread across two adjacent chunks of the same document. The decision
is returned as telemetry (coverage, matched terms, threshold) so the API can
show the caller exactly why it answered or refused.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

from ..index.bm25 import tokenize
from ..types import Hit


@dataclass
class Grounding:
    grounded: bool
    coverage: float
    threshold: float
    query_terms: List[str]
    matched_terms: List[str]
    passages_checked: int

    def as_dict(self) -> Dict:
        return asdict(self)


def check_grounding(question: str, hits: Sequence[Hit], threshold: float) -> Grounding:
    """Return the grounding decision for `question` against `hits`.

    Coverage = |query terms present in any top passage| / |query terms|.
    A question whose terms are absent everywhere scores 0.0 and is refused.
    """
    q_terms = list(dict.fromkeys(tokenize(question)))
    seen: set = set()
    for hit in hits:
        seen.update(tokenize(hit.chunk.text))
    matched = [term for term in q_terms if term in seen]
    coverage = len(matched) / len(q_terms) if q_terms else 0.0
    return Grounding(
        grounded=bool(q_terms) and coverage >= threshold,
        coverage=round(coverage, 4),
        threshold=threshold,
        query_terms=q_terms,
        matched_terms=matched,
        passages_checked=len(hits),
    )
