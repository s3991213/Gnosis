"""
Document loader for Gnosis.

Supports loading content from:
- Markdown files (.md, .markdown)
- PDF files (.pdf) via PyMuPDF
- Plain text files (.txt)
- Web URLs via httpx + BeautifulSoup

Each loader returns a list of ``Document`` objects carrying the raw text
and source metadata (file path or URL, MIME type, page number where
applicable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Document:
    """Raw document returned by a loader before chunking."""

    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class Loader(Protocol):
    """Common interface every loader must satisfy."""

    def load(self, source: str) -> list[Document]:
        """Load *source* and return one or more Documents."""
        ...


class MarkdownLoader:
    """Load a single Markdown file into a Document."""

    def load(self, source: str) -> list[Document]:
        text = Path(source).read_text(encoding="utf-8")
        return [Document(content=text, source=source, metadata={"type": "markdown"})]


class PDFLoader:
    """Load a PDF file page-by-page using PyMuPDF."""

    def load(self, source: str) -> list[Document]:
        import fitz  # PyMuPDF

        docs = []
        with fitz.open(source) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    docs.append(
                        Document(
                            content=text,
                            source=source,
                            metadata={"type": "pdf", "page": page_num},
                        )
                    )
        return docs


class TextLoader:
    """Load a plain-text file into a Document."""

    def load(self, source: str) -> list[Document]:
        text = Path(source).read_text(encoding="utf-8")
        return [Document(content=text, source=source, metadata={"type": "text"})]


class URLLoader:
    """Fetch a web URL and return its visible text as a Document."""

    def load(self, source: str) -> list[Document]:
        import httpx
        from bs4 import BeautifulSoup

        response = httpx.get(source, follow_redirects=True, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return [Document(content=text, source=source, metadata={"type": "url"})]


_SUFFIX_MAP: dict[str, Loader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".txt": TextLoader(),
    ".pdf": PDFLoader(),
}


def load(source: str) -> list[Document]:
    """Auto-detect the type of *source* and delegate to the right loader."""
    if source.startswith("http://") or source.startswith("https://"):
        return URLLoader().load(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if path.is_dir():
        docs: list[Document] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in _SUFFIX_MAP:
                docs.extend(load(str(child)))
        return docs

    loader = _SUFFIX_MAP.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader.load(source)
