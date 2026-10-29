from __future__ import annotations

from caracal_rag.chunking import chunk_document, chunk_markdown
from caracal_rag.sources import Document


def test_markdown_starting_with_heading_has_no_empty_leading_chunk() -> None:
    content = "# Login\n\nDetails here.\n\n## Expiration\n\nTokens expire."

    chunks = chunk_markdown(content)

    assert len(chunks) == 1
    assert chunks[0].startswith("# Login")
    assert "## Expiration" in chunks[0]
    assert all(chunk.strip() for chunk in chunks)
    assert "".join(chunks) == content


def test_markdown_groups_short_sections_without_exceeding_target() -> None:
    content = "# A\n\nalpha\n\n## B\n\nbravo\n\n## C\n\ncharlie"

    chunks = chunk_markdown(content, max_chars=100, target_chars=25)

    assert len(chunks) == 2
    assert chunks[0].startswith("# A")
    assert "## B" in chunks[0]
    assert chunks[1].startswith("## C")
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert "".join(chunks) == content


def test_adjacent_short_sections_that_cross_target_stay_together() -> None:
    content = (
        "## Reducer\n\n"
        "reducer updates state.\n\n"
        "## Side effects\n\n"
        "sideEffect records logs.\n\n"
        "## Other\n\n"
        "other info"
    )

    chunks = chunk_markdown(content, max_chars=100, target_chars=60)

    assert len(chunks) == 2
    assert "reducer" in chunks[0]
    assert "sideEffect" in chunks[0]
    assert "## Other" in chunks[1]
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks) == content


def test_markdown_preserves_larger_sections_below_maximum() -> None:
    first_section = "# Details\n\n" + ("x" * 50) + "\n\n"
    second_section = "## More\n\nshort"
    content = first_section + second_section

    chunks = chunk_markdown(content, max_chars=100, target_chars=30)

    assert chunks == [first_section, second_section]
    assert "".join(chunks) == content


def test_markdown_splits_oversized_sections_without_losing_text() -> None:
    content = "# Intro\n\nShort.\n\n## Details\n\n" + ("detail text " * 15)

    chunks = chunk_markdown(content, max_chars=40, target_chars=20)

    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunks) == content


def test_chunk_document_indexes_markdown_chunks_from_zero() -> None:
    content = (
        "# Login\n\n"
        + ("l" * 410)
        + "\n\n## Expiration\n\n"
        + ("e" * 410)
    )
    document = Document(
        source="api-docs",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content=content,
        content_hash=Document.hash(content),
    )

    chunks = list(chunk_document(document))

    assert len(chunks) == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.text.strip() for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content
