"""Document loading: markdown / text / PDF -> Document with sections.

Sections come from markdown headings. For PDFs we fall back to page
boundaries, which gives the same citation granularity ("page 4").
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ..types import Document, Section

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.M)


def _doc_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "-")


def _title_from(text: str, path: Path) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        return m.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = _title_from(text, path)

    # Split on headings of level 2-4; keep the H1 as the document title.
    marks = [(m.start(), m.end(), m.group(2).strip()) for m in _HEADING.finditer(text)
             if len(m.group(1)) >= 2]

    sections: List[Section] = []
    if not marks:
        body = text.strip()
        if body:
            sections.append(Section(heading="Overview", text=body, order=0))
    else:
        for i, (start, end, heading) in enumerate(marks):
            stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            body = text[end:stop].strip()
            if body:
                sections.append(Section(heading=heading, text=body, order=len(sections)))

    return Document(doc_id=_doc_id(path), title=title, path=str(path), sections=sections)


def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: List[Section] = []
    for i, page in enumerate(reader.pages):
        try:
            body = (page.extract_text() or "").strip()
        except Exception:
            body = ""
        if body:
            sections.append(Section(heading="page %d" % (i + 1), text=body, order=i))
    return Document(doc_id=_doc_id(path), title=path.stem, path=str(path), sections=sections)


def load_plain(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    sections = [Section(heading="Overview", text=text, order=0)] if text else []
    return Document(doc_id=_doc_id(path), title=_title_from(text, path),
                    path=str(path), sections=sections)


SUPPORTED = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".txt": load_plain,
}


def load_document(path: Path) -> Document:
    loader = SUPPORTED.get(path.suffix.lower())
    if loader is None:
        raise ValueError("unsupported file type: %s" % path)
    return loader(path)


def load_corpus(corpus_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            doc = load_document(path)
            if doc.sections:
                docs.append(doc)
    return docs
