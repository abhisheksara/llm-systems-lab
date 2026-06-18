from __future__ import annotations
import hashlib, re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_html(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def chunk_document(text: str, metadata: dict, max_tokens: int = 512) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = len(enc.encode(sent))
        if st > max_tokens:
            if buf:
                _flush(buf, metadata, chunks)
                buf, buf_tokens = [], 0
            words, sub = sent.split(), []
            for w in words:
                candidate = " ".join(sub + [w])
                if len(enc.encode(candidate)) > max_tokens and sub:
                    _flush(sub, metadata, chunks); sub = []
                sub.append(w)
            if sub:
                _flush(sub, metadata, chunks)
            continue
        if len(enc.encode(" ".join(buf + [sent]))) > max_tokens and buf:
            _flush(buf, metadata, chunks); buf, buf_tokens = [], 0
        buf.append(sent); buf_tokens += st

    if buf:
        _flush(buf, metadata, chunks)
    return chunks


def _flush(parts: list[str], metadata: dict, out: list[Chunk]) -> None:
    text = " ".join(parts)
    cid = hashlib.md5(text.encode()).hexdigest()[:12]
    out.append(Chunk(text=text, metadata={**metadata, "chunk_id": cid}, chunk_id=cid))
