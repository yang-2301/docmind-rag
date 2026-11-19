"""Retrieval evaluation.

Measures the four pipeline configurations on the same golden set, so a
change to chunking, fusion or reranking shows up as a number instead of
an opinion.

Metrics
  recall@k : at least one gold document appears in the top-k
  mrr@10   : mean reciprocal rank of the first correct hit
  p50/p95  : end-to-end retrieval latency per query

Usage
  python3 -m docmind.eval.run_eval                     # markdown table
  python3 -m docmind.eval.run_eval --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

from ..config import settings
from ..service import MODES, DocMind


def load_golden(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(service: DocMind, questions: Sequence[dict], mode: str,
             depth: int = 10) -> Dict[str, float]:
    latencies: List[float] = []
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal: List[float] = []
    misses: List[dict] = []

    for row in questions:
        started = time.perf_counter()
        hits = service.search(row["question"], mode=mode, top_k=depth)
        latencies.append((time.perf_counter() - started) * 1000)

        gold = set(row["gold"])
        rank = None
        for i, hit in enumerate(hits, start=1):
            if hit.chunk.doc_id in gold:
                rank = i
                break

        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        reciprocal.append(1.0 / rank if rank else 0.0)
        if rank is None:
            misses.append({"question": row["question"], "gold": row["gold"]})

    n = len(questions) or 1
    latencies_sorted = sorted(latencies)
    p95_index = min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)))
    return {
        "mode": mode,
        "recall@1": hits_at[1] / n,
        "recall@3": hits_at[3] / n,
        "recall@5": hits_at[5] / n,
        "recall@10": hits_at[10] / n,
        "mrr@10": sum(reciprocal) / n,
        "p50_ms": statistics.median(latencies),
        "p95_ms": latencies_sorted[p95_index],
        "misses": misses,
    }


def by_type(service: DocMind, questions: Sequence[dict], mode: str,
            depth: int = 10) -> Dict[str, Dict[str, float]]:
    """Recall broken down by question type.

    A single aggregate number hides the only thing that matters when you
    are choosing a retriever: which failure mode you are buying.
    """
    buckets: Dict[str, List[dict]] = {}
    for row in questions:
        buckets.setdefault(row.get("type", "other"), []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    for name, rows in sorted(buckets.items()):
        stats = evaluate(service, rows, mode, depth=depth)
        out[name] = {
            "n": len(rows),
            "recall@1": stats["recall@1"],
            "recall@5": stats["recall@5"],
            "mrr@10": stats["mrr@10"],
        }
    return out


def type_table(per_mode: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    types = sorted({t for stats in per_mode.values() for t in stats})
    modes = list(per_mode)
    header = "| question type | n | " + " | ".join("`%s` R@1" % m for m in modes) + " |"
    sep = "|---|---|" + "---|" * len(modes)
    lines = [header, sep]
    for t in types:
        n = next(per_mode[m][t]["n"] for m in modes if t in per_mode[m])
        cells = []
        for m in modes:
            val = per_mode[m].get(t, {}).get("recall@1")
            cells.append("%.0f%%" % (100 * val) if val is not None else "—")
        lines.append("| %s | %d | %s |" % (t, n, " | ".join(cells)))
    return "\n".join(lines)


def markdown_table(results: List[Dict[str, float]], n_questions: int) -> str:
    head = ("| mode | recall@1 | recall@3 | recall@5 | MRR@10 | p50 (ms) | p95 (ms) |\n"
            "|---|---|---|---|---|---|---|")
    rows = [
        "| `%s` | %.1f%% | %.1f%% | %.1f%% | %.3f | %.1f | %.1f |" % (
            r["mode"], 100 * r["recall@1"], 100 * r["recall@3"], 100 * r["recall@5"],
            r["mrr@10"], r["p50_ms"], r["p95_ms"],
        )
        for r in results
    ]
    return "\n".join([head] + rows) + "\n\n_%d questions, single run, local index._" % n_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(settings.root / "data" / "golden.jsonl"))
    parser.add_argument("--json", default="")
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()

    questions = load_golden(Path(args.golden))
    service = DocMind.load(settings)

    results = [evaluate(service, questions, mode, depth=max(10, settings.top_k))
               for mode in args.modes.split(",")]

    print("\nIndex: %d chunks, %d documents, embedder=%s\n"
          % (service.index.size, len(service.documents), service.index.embedder.name))
    print(markdown_table(results, len(questions)))

    per_mode = {}
    for mode in args.modes.split(","):
        per_mode[mode] = by_type(service, questions, mode, depth=max(10, settings.top_k))

    print("\nRecall@1 by question type\n")
    print(type_table(per_mode))

    for r in results:
        if r["misses"]:
            print("\n%s missed %d:" % (r["mode"], len(r["misses"])))
            for m in r["misses"][:6]:
                print("  - %s  (gold: %s)" % (m["question"], ", ".join(m["gold"])))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
