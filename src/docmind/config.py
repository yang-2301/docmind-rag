"""Runtime configuration. Everything has a working default so the
project runs with zero setup."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    corpus_dir: Path = ROOT / "data" / "corpus"
    index_dir: Path = ROOT / "data" / "index"
    db_path: Path = ROOT / "data" / "index" / "docmind.db"

    # chunking
    chunk_size: int = 900          # characters
    chunk_overlap: int = 180

    # retrieval
    top_k: int = int(os.getenv("DOCMIND_TOP_K", "8"))
    candidate_k: int = 40          # per-retriever depth before fusion
    rrf_k: int = 60
    rerank: bool = _flag("DOCMIND_RERANK")

    # grounding gate — fraction of the question's content terms that must
    # appear in the union of the top passages before we answer at all.
    # 0.30 separates the bundled golden set (min 0.33) from unrelated
    # questions (max 0.25). Lower it only after re-running `make eval`.
    grounding_min_coverage: float = float(os.getenv("DOCMIND_MIN_COVERAGE", "0.30"))

    # embeddings
    embedder: str = os.getenv("DOCMIND_EMBEDDER", "lsa").strip().lower()
    embed_model: str = os.getenv("DOCMIND_EMBED_MODEL", "text-embedding-3-small")
    lsa_dim: int = 128

    # generation
    llm_model: str = os.getenv("DOCMIND_LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
