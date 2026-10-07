"""
Question answering over the knowledge base.

Retrieves the passages most relevant to a question (hybrid search + rerank),
numbers them, and asks the local LLM to answer *only* from those passages,
citing them inline as ``[1]``, ``[2]``. Follow-up questions carry the recent
conversation so "what about X?" is understood in context.

Usage::

    sources = find_sources("What is a rational agent?")
    for text in stream_answer("What is a rational agent?", sources):
        ...  # text is the answer so far
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Cross-encoder (ms-marco) score below which a passage is treated as unrelated.
# On the sample notes, real matches score above about -7 and unrelated
# notes cluster between -10 and -11.5.
WEAK_SCORE = -8.0

# Earlier turns sent with a follow-up; older ones are dropped to keep the
# prompt within a small model's context.
_HISTORY_TURNS = 3

_SYSTEM_PROMPT = """You answer questions using numbered passages from the user's personal \
knowledge base (their notes, documents and books).

Rules:
- Use only the passages. Do not add facts from your own knowledge.
- Cite the passages each statement comes from, inline, like [1] or [2][3].
- If the passages don't answer the question, say so plainly instead of guessing.
- Passages extracted from PDFs may contain stray page numbers, headers, margin \
notes or hyphenated line breaks; ignore those.
- Be clear and well organised. Use Markdown where it helps."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


@dataclass
class Turn:
    """One question and the answer given to it."""

    question: str
    answer: str


def source_label(result: dict) -> str:
    """Readable name for a result's source: a filename, URL or note title, plus PDF page."""
    meta = result.get("metadata", {})
    source = meta.get("source") or result.get("id", "unknown")
    if source.startswith("note:"):
        name = f"Untitled note ({source[5:11]})"
    else:
        name = Path(source).name if source.startswith("/") else source
    if meta.get("page"):
        name += f", p. {meta['page']}"
    return name


def find_sources(question: str, history: Sequence[Turn] = (), n: int = 5) -> list[dict]:
    """
    Return up to *n* passages relevant to *question*, best first.

    A follow-up is searched together with the previous question, since on its
    own ("and its drawbacks?") it rarely names the subject. Passages the
    reranker scores as unrelated are dropped, so the list may be empty.
    """
    from gnosis.config import get_reranker, get_searcher

    query = f"{history[-1].question} {question}" if history else question
    candidates = get_searcher().search(query, n_results=max(n * 4, 20))
    if not candidates:
        return []
    ranked = get_reranker().rerank(query, candidates, top_n=n)
    return [r for r in ranked if r.get("rerank_score", 0.0) > WEAK_SCORE]


def build_messages(
    question: str, sources: Sequence[dict], history: Sequence[Turn] = ()
) -> list[dict]:
    """Chat messages asking the LLM to answer *question* from the numbered *sources*."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history[-_HISTORY_TURNS:]:
        # Old citation numbers refer to old passages; drop them.
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": _CITATION.sub("", turn.answer)})

    passages = "\n\n".join(
        f"[{i}] ({source_label(s)})\n{s['text'].strip()}" for i, s in enumerate(sources, start=1)
    )
    # Rules are repeated after the passages: small models follow instructions
    # next to the question much more reliably than ones in the system prompt.
    messages.append(
        {
            "role": "user",
            "content": f"Passages:\n\n{passages}\n\nQuestion: {question}\n\n"
            "Answer using only the passages above and cite each statement like [1]. "
            "If the passages don't answer it, say so.",
        }
    )
    return messages


def clean_answer(text: str) -> str:
    """Remove reasoning blocks (``<think>…</think>``) some models emit before answering."""
    return _THINK_BLOCK.sub("", text).strip()


def stream_answer(
    question: str, sources: Sequence[dict], history: Sequence[Turn] = ()
) -> Iterator[str]:
    """Yield the answer so far (reasoning blocks removed) as the LLM writes it."""
    from gnosis.config import get_llm_client

    raw = ""
    for delta in get_llm_client().chat_stream(build_messages(question, sources, history)):
        raw += delta
        yield clean_answer(raw)


def answer(question: str, sources: Sequence[dict], history: Sequence[Turn] = ()) -> str:
    """Return the full answer in one call (no streaming)."""
    from gnosis.config import get_llm_client

    reply = get_llm_client().chat(build_messages(question, sources, history))
    return clean_answer(reply)


def cited(answer_text: str, sources: Sequence[dict]) -> list[tuple[int, dict]]:
    """``(number, source)`` pairs the answer cites, in number order."""
    numbers: set[int] = set()
    for group in _CITATION.findall(answer_text):
        numbers.update(int(n) for n in group.split(","))
    return [(i, s) for i, s in enumerate(sources, start=1) if i in numbers]
