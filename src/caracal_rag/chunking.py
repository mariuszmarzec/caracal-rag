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


def _markdown_blocks(section: str) -> list[str]:
    """Split a Markdown section into paragraphs while keeping fenced code atomic."""
    lines = section.splitlines(keepends=True)
    blocks: list[str] = []
    current: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    for line in lines:
        current.append(line)
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0

        if fence_character is None and not line.strip():
            blocks.append("".join(current))
            current.clear()

    if current:
        blocks.append("".join(current))

    # Keep a heading with the first paragraph it introduces instead of creating
    # a tiny title-only chunk.
    merged: list[str] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        non_empty_lines = [line for line in block.splitlines() if line.strip()]
        heading_only = (
            len(non_empty_lines) == 1
            and re.match(r"^#{1,6}[ \t]+", non_empty_lines[0]) is not None
        )
        if heading_only and index + 1 < len(blocks):
            merged.append(block + blocks[index + 1])
            index += 2
        else:
            merged.append(block)
            index += 1
    return merged


def _chunk_code_section(section: str, max_chars: int, target_chars: int) -> list[str]:
    """Chunk a fenced-code Markdown section by prose/code blocks.

    Paragraph and fenced-code boundaries preserve coherent retrieval context.
    Code fences stay atomic unless an individual block exceeds the hard limit.
    """
    chunks: list[str] = []
    buffered_blocks: list[str] = []
    buffered_length = 0

    def flush_buffer() -> None:
        nonlocal buffered_length
        if buffered_blocks:
            chunks.append("".join(buffered_blocks))
            buffered_blocks.clear()
            buffered_length = 0

    for block in _markdown_blocks(section):
        if len(block) > max_chars:
            flush_buffer()
            chunks.extend(
                block[i : i + max_chars]
                for i in range(0, len(block), max_chars)
            )
        elif len(block) >= target_chars:
            flush_buffer()
            chunks.append(block)
        else:
            if buffered_blocks and buffered_length + len(block) > max_chars:
                flush_buffer()
            buffered_blocks.append(block)
            buffered_length += len(block)
            if buffered_length >= target_chars:
                flush_buffer()

    flush_buffer()
    return chunks or [section]


def chunk_markdown(
    text: str,
    max_chars: int = 1800,
    target_chars: int = 400,
) -> list[str]:
    """Chunk Markdown by headings and paragraphs, preserving fenced code blocks."""
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
        elif len(section) >= target_chars and re.search(
            r"^[ \t]*(?:`{3,}|~{3,})", section, re.MULTILINE
        ):
            flush_buffer()
            chunks.extend(_chunk_code_section(section, max_chars, target_chars))
        elif len(section) >= target_chars:
            flush_buffer()
            chunks.append(section)
        else:
            # Preserve adjacent short sections together when the next section
            # crosses the target. The target is a soft boundary; max_chars is
            # the hard limit. Flush before adding only when the hard limit
            # would otherwise be exceeded.
            if buffered_sections and buffered_length + len(section) > max_chars:
                flush_buffer()
            buffered_sections.append(section)
            buffered_length += len(section)
            if buffered_length >= target_chars:
                flush_buffer()

    flush_buffer()
    return chunks or [text]


def chunk_document(document: Document, max_chars: int = 1800) -> Iterable[Chunk]:
    chunks: list[str]
    if document.type == "markdown":
        chunks = chunk_markdown(document.content, max_chars=max_chars)
    else:
        content = document.content
        chunks = [
            content[i : i + max_chars]
            for i in range(0, len(content), max_chars)
        ] or [content]

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
