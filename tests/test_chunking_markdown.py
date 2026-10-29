from __future__ import annotations

from caracal_rag.chunking import chunk_document, chunk_markdown
from caracal_rag.sources import Document


def test_markdown_starting_with_heading_has_no_empty_leading_chunk() -> None:
    content = "# Login\n\nDetails here.\n\n## Expiration\n\nTokens expire."

    chunks = chunk_markdown(content)

    assert len(chunks) == 2
    assert chunks[0].startswith("# Login")
    assert all(chunk.strip() for chunk in chunks)
    assert "".join(chunks) == content


def test_chunk_document_indexes_non_empty_markdown_chunks_from_zero() -> None:
    content = "# Login\n\nDetails here.\n\n## Expiration\n\nTokens expire."
    document = Document(
        source="api-docs",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content=content,
        content_hash=Document.hash(content),
    )

    chunks = list(chunk_document(document))

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text.strip() for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content
