from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from caracal_rag.sources import Document


@dataclass
class Chunk:
    source: str
    document: str
    url: str
    type: str
    content_hash: str
    chunk_index: int
    text: str


def infer_type(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".md"):
        return "markdown"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return "yaml"
    return "text"


def _md_sections(markdown: str) -> Iterable[tuple[int, int]]:
    headings = re.finditer(r"^#{1,6} .*$", markdown, re.MULTILINE)
    # A heading at offset 0 duplicates the initial boundary. De-duplicate
    # boundaries so the first section is the real content, not an empty chunk.
    positions = sorted({0, *(match.start() for match in headings), len(markdown)})
    return [(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]


def chunk_markdown(
    text: str,
    max_chars: int = 1800,
    target_chars: int = 400,
) -> list[str]:
    """Chunk Markdown by headings, grouping adjacent short sections together."""
    target_chars = min(target_chars, max_chars)
    chunks: list[str] = []
    buffered_sections: list[str] = []
    buffered_length = 0

    def flush_buffer() -> None:
        nonlocal buffered_length
        if buffered_sections:
            chunks.append("".join(buffered_sections))
            buffered_sections.clear()
            buffered_length = 0

    for start, end in _md_sections(text):
        section = text[start:end]
        if len(section) > max_chars:
            flush_buffer()
            chunks.extend(
                section[i : i + max_chars]
                for i in range(0, len(section), max_chars)
            )
        elif len(section) >= target_chars:
            flush_buffer()
            chunks.append(section)
        else:
            if buffered_sections and buffered_length + len(section) > target_chars:
                flush_buffer()
            buffered_sections.append(section)
            buffered_length += len(section)

    flush_buffer()
    return chunks or [text]


def chunk_document(document: Document, max_chars: int = 1800) -> Iterable[Chunk]:
    chunks: list[str]
    if document.type == "markdown":
        chunks = chunk_markdown(document.content, max_chars=max_chars)
    else:
        content = document.content
        chunks = [content[i : i + max_chars] for i in range(0, len(content), max_chars)] or [content]

    for idx, text in enumerate(chunks):
        yield Chunk(
            source=document.source,
            document=document.name,
            url=document.url,
            type=document.type,
            content_hash=document.content_hash,
            chunk_index=idx,
            text=text,
        )
