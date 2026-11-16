"""Load documents and split them into section-aware chunks with contextual headers."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    doc: str  # human-readable document name
    section: str
    text: str  # the raw passage, shown to users and quoted in answers
    page: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        """Contextual chunk header: the document and section are indexed with the passage, so a chunk that
        only says "the rear 40% of the plot" can still be found by a question about "the Malabar property"."""
        return f"{self.doc} | {self.section}\n{self.text}"


@dataclass
class Document:
    name: str
    pages: list[str]  # one entry per page (a single entry for text and web pages)
    fingerprint: str


# Headings like "## 2. Payment plan" or "3. PENDING LITIGATION AND DISPUTES" start a new section.
_HEADING = re.compile(r"^(?:#{1,3}\s+(.+)|(\d{1,2}\.\s+[A-Z][A-Za-z ,&/-]{3,60}))$", re.MULTILINE)
_INLINE_HEADING = re.compile(r"(?<!\S)(\d{1,2}\.\s+[A-Z][A-Z ,&/-]{3,60}[A-Z])(?=\s)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def pretty_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").strip()


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _clean(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def load_pdf(name: str, data: bytes) -> Document:
    reader = PdfReader(io.BytesIO(data))
    pages = [_clean(page.extract_text() or "") for page in reader.pages]
    return Document(pretty_name(name), pages, _fingerprint(data))


def load_text(name: str, data: bytes) -> Document:
    return Document(pretty_name(name), [_clean(data.decode("utf-8", errors="replace"))], _fingerprint(data))


def load_url(url: str, timeout: int = 15) -> Document:
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "satyori-rag/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = (soup.title.string or url).strip() if soup.title else url
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return Document(title[:80], [_clean(text)], _fingerprint(resp.content))


def load_file(path: Path) -> Document:
    data = path.read_bytes()
    return load_pdf(path.name, data) if path.suffix.lower() == ".pdf" else load_text(path.name, data)


def _sections(page: str) -> list[tuple[str, str]]:
    """Split a page into (heading, body) pairs. Text before the first heading is the 'Overview'."""
    page = _INLINE_HEADING.sub(r"\n\1\n", page)  # PDFs often flatten headings into the running text
    marks = list(_HEADING.finditer(page))
    if not marks:
        return [("Overview", page)]
    out = []
    if page[: marks[0].start()].strip():
        out.append(("Overview", page[: marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(page)
        heading = (m.group(1) or m.group(2)).strip().lstrip("#").strip()
        out.append((heading.title() if heading.isupper() else heading, page[m.end() : end]))
    return out


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedy sentence packing: chunks never cut a sentence, and share `overlap` sentences with the next."""
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, size: int = 700, overlap: int = 1) -> list[Chunk]:
    # Sections are cut from the whole document, so a sentence that runs over a page break stays in one piece.
    full = "\n".join(doc.pages)
    flat_pages = [" ".join(p.split()) for p in doc.pages]
    chunks: list[Chunk] = []
    for heading, body in _sections(full):
        sentences = [s.strip() for s in _SENTENCE.split(" ".join(body.split())) if s.strip()]
        for text in _pack(sentences, size, overlap):
            if len(text) < 40 and chunks and chunks[-1].section == heading:
                chunks[-1].text += " " + text  # fold tiny tails into the previous chunk
                continue
            page = next((n for n, p in enumerate(flat_pages, 1) if text[:40] in p), None)
            cid = f"{doc.fingerprint}-{len(chunks):03d}"
            chunks.append(Chunk(cid, doc.name, heading, text, page if len(doc.pages) > 1 else None))
    return chunks
