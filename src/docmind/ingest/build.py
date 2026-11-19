"""CLI entry point: ingest the corpus into the local index.

  python3 -m docmind.ingest.build                 # default corpus
  python3 -m docmind.ingest.build --path ./docs   # your own folder
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config import settings
from ..index.pipeline import build_index
from ..index.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DocMind index")
    parser.add_argument("--path", default=str(settings.corpus_dir),
                        help="folder of .md / .txt / .pdf documents")
    args = parser.parse_args()

    started = time.perf_counter()
    store = Store(settings.db_path)
    index = build_index(Path(args.path), settings, store)
    elapsed = time.perf_counter() - started

    docs = len({c.doc_id for c in index.chunks})
    print("indexed %d documents -> %d chunks in %.2fs" % (docs, index.size, elapsed))
    print("embedder : %s (dim=%d)" % (index.embedder.name, index.embedder.dim))
    print("store    : %s" % settings.db_path)


if __name__ == "__main__":
    main()
