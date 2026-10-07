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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Control characters other than tab/newline/carriage return. PDF text
# extraction can emit these (NUL in particular), and Postgres rejects NUL.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
        import pymupdf

        docs = []
        with pymupdf.open(source) as pdf:
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

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and not (
            content_type.startswith("text/") or content_type.endswith(("html", "xml"))
        ):
            raise ValueError(
                f"That link is a file ({content_type}), not a web page. "
                "Download it and import the file instead."
            )

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


def expand(source: str) -> list[str]:
    """
    Resolve *source* to the individual files/URLs to load.

    A directory becomes every supported file under it; a URL or file is
    returned as-is. Lets callers show per-file progress.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return [source]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if path.is_dir():
        return [
            str(child)
            for child in sorted(path.rglob("*"))
            if child.is_file() and child.suffix.lower() in _SUFFIX_MAP
        ]

    if path.suffix.lower() not in _SUFFIX_MAP:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return [source]


def load(source: str) -> list[Document]:
    """Auto-detect the type of *source* and delegate to the right loader."""
    docs: list[Document] = []
    for item in expand(source):
        if item.startswith("http://") or item.startswith("https://"):
            docs.extend(URLLoader().load(item))
        else:
            docs.extend(_SUFFIX_MAP[Path(item).suffix.lower()].load(item))
    for doc in docs:
        doc.content = _CONTROL_CHARS.sub("", doc.content)
    return docs
