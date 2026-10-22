"""Tests for the indexing pipeline: idempotency, change, and deletion behaviour.

External services (LiteLLM, GitHub, Chroma) are replaced with in-memory fakes so
the tests run without network access.

The :class:`FakeCollection` faithfully honours the Chroma ``include`` argument:
when ``metadatas`` is not requested it is omitted from the result, mirroring the
real Chroma behaviour that motivated the ``delete_missing_documents`` fix.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Stub chromadb so the tests run without a working Chroma server or a
# compatible sqlite3 build; the real client is replaced by FakeCollection via
# monkeypatch in each test.
_chromadb_stub = types.ModuleType("chromadb")
_chromadb_stub.HttpClient = object
sys.modules["chromadb"] = _chromadb_stub

from caracal_rag import chunking  # noqa: E402
from caracal_rag.config import AppConfig, SourceConfig  # noqa: E402
from caracal_rag.sources import Document, SourceLoadError  # noqa: E402
from caracal_rag.vectorstore import (  # noqa: E402
    _chunk_id,
    assert_embedding_model_consistent,
    search_chunks,
)


class FakeCollection:
    """Minimal in-memory Chroma collection that honours the ``include`` argument."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(
        self,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        items = list(self._store.items())
        if where is not None:
            url = where.get("url")
            items = [(cid, meta) for cid, meta in items if meta.get("url") == url]

        result: dict[str, Any] = {"ids": [cid for cid, _ in items]}
        if include is not None:
            if "metadatas" in include:
                result["metadatas"] = [record.get("metadata", record) for _, record in items]
            if "documents" in include:
                result["documents"] = [record.get("text", "") for _, record in items]
            if "embeddings" in include:
                result["embeddings"] = [record.get("embedding") for _, record in items]
        return result

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for index, (cid, text, meta) in enumerate(zip(ids, documents, metadatas)):
            self._store[cid] = {
                "text": text,
                "embedding": embeddings[index] if index < len(embeddings) else None,
                "metadata": meta,
                **meta,
            }

    def add(self, **kwargs) -> None:
        self.upsert(**kwargs)

    def delete(self, ids) -> None:
        for cid in ids:
            self._store.pop(cid, None)

    def query(self, query_embeddings, n_results: int = 5) -> dict[str, Any]:
        # Stage rows sort first so tests prove search filters them.
        items = sorted(
            self._store.items(),
            key=lambda item: (
                not item[1].get("metadata", {}).get("caracal_staging", False),
                item[0],
            ),
        )[:n_results]
        return {
            "ids": [[cid for cid, _ in items]],
            "documents": [[record.get("text", "") for _, record in items]],
            "metadatas": [[record.get("metadata", {}) for _, record in items]],
            "distances": [[0.0 for _ in items]],
        }


def _make_config() -> AppConfig:
    return AppConfig(
        sources=[SourceConfig(name="api", type="markdown", urls=["https://example.com/login.md"])],
        embedding=AppConfig.from_yaml("config/sources.example.yaml").embedding,
        chroma=AppConfig.from_yaml("config/sources.example.yaml").chroma,
        config_path=Path("config/sources.example.yaml"),
    )


def _patch_embed(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    def fake_embed(texts, api_base, api_key=None, model=None):
        captured["texts"] = list(texts)
        captured["api_base"] = api_base
        captured["model"] = model
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("caracal_rag.indexing.embed_texts", fake_embed)


def _make_markdown_doc(source: str, name: str, url: str, content: str) -> Document:
    return Document(
        source=source,
        name=name,
        url=url,
        type="markdown",
        content=content,
        content_hash=Document.hash(content),
    )


def _stored_models(collection: FakeCollection) -> set[str]:
    models = set()
    for meta in collection._store.values():
        model = meta.get("embedding_model")
        if model:
            models.add(model)
    return models


def _patch_documents(monkeypatch: pytest.MonkeyPatch, documents: list[Document]) -> None:
    monkeypatch.setattr(
        "caracal_rag.indexing.load_documents_from_config",
        lambda path: list(documents),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, collection: FakeCollection) -> FakeCollection:
    class _FakeChroma:
        def get_or_create_collection(self, name: str) -> FakeCollection:
            return collection

    monkeypatch.setattr(
        "caracal_rag.indexing._chroma_client",
        lambda *args, **kwargs: _FakeChroma(),
    )
    return collection


def _stored_urls(collection: FakeCollection) -> set[str]:
    return {meta.get("url") for meta in collection._store.values()}


def test_repeated_index_on_unchanged_docs_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from caracal_rag.indexing import Indexer

    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    _patch_documents(monkeypatch, [doc])
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    indexer = Indexer(config=_make_config())
    indexer.run()
    first_count = len(collection._store)

    # Second run: document unchanged -> no new embeddings, no duplicate vectors.
    captured.clear()
    indexer.run()
    assert "texts" not in captured  # nothing was re-embedded
    assert len(collection._store) == first_count


def test_changed_doc_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    from caracal_rag.indexing import Indexer

    original = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nOld details.",
        content_hash=Document.hash("# Login\n\nOld details."),
    )
    updated = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nNew details.",
        content_hash=Document.hash("# Login\n\nNew details."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    _patch_documents(monkeypatch, [original])
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    indexer = Indexer(config=_make_config())
    indexer.run()
    assert captured["texts"] != []

    # Change the document and re-index.
    captured.clear()
    _patch_documents(monkeypatch, [updated])
    indexer.run()
    assert captured["texts"] != []  # changed doc was re-embedded
    # Old chunks removed, new ones present; no stale vectors for this url.
    for cid, meta in collection._store.items():
        assert meta.get("content_hash") == updated.content_hash


def test_deleted_doc_removes_stale_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from caracal_rag.indexing import Indexer

    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    _patch_documents(monkeypatch, [doc])
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    config = _make_config()
    # This test intentionally simulates a complete, legitimate empty inventory.
    config.sources[0].allow_empty = True
    indexer = Indexer(config=config)
    indexer.run()
    assert len(collection._store) > 0

    # Document is gone from the source on the next run.
    captured.clear()
    _patch_documents(monkeypatch, [])
    indexer.run()
    assert len(collection._store) == 0


def test_indexing_skips_empty_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    from caracal_rag.indexing import Indexer

    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    _patch_documents(monkeypatch, [])
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    indexer = Indexer(config=_make_config())
    indexer.run()
    assert "texts" not in captured  # no documents -> no embeddings requested
    assert len(collection._store) == 0


def test_indexing_one_source_does_not_delete_other_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: indexing a single source must not delete another source's vectors.

    Running ``index --source fiteo-docs`` previously fed only that source's URLs
    into a collection-wide ``delete_missing_documents``, which wiped out chunks
    belonging to every other configured source.
    """
    from caracal_rag.indexing import Indexer

    doc_a = Document(
        source="source-a",
        name="a.md",
        url="https://example.com/a.md",
        type="markdown",
        content="# A\n\nSource A content.",
        content_hash=Document.hash("# A\n\nSource A content."),
    )
    doc_b = Document(
        source="source-b",
        name="b.md",
        url="https://example.com/b.md",
        type="markdown",
        content="# B\n\nSource B content.",
        content_hash=Document.hash("# B\n\nSource B content."),
    )

    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    # First index source A.
    _patch_documents(monkeypatch, [doc_a])
    Indexer(config=_make_config(), source_filter="source-a").run()
    assert _stored_urls(collection) == {doc_a.url}

    # Now index source B. Source A's vectors must be preserved.
    captured.clear()
    _patch_documents(monkeypatch, [doc_b])
    Indexer(config=_make_config(), source_filter="source-b").run()
    assert _stored_urls(collection) == {doc_a.url, doc_b.url}


def test_full_index_still_reconciles_all_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full (unfiltered) run still removes genuinely stale chunks across sources."""
    from caracal_rag.indexing import Indexer

    doc_a = Document(
        source="source-a",
        name="a.md",
        url="https://example.com/a.md",
        type="markdown",
        content="# A",
        content_hash=Document.hash("# A"),
    )
    doc_b = Document(
        source="source-b",
        name="b.md",
        url="https://example.com/b.md",
        type="markdown",
        content="# B",
        content_hash=Document.hash("# B"),
    )

    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    _patch_documents(monkeypatch, [doc_a, doc_b])
    Indexer(config=_make_config()).run()
    assert _stored_urls(collection) == {doc_a.url, doc_b.url}

    # Full run again but doc_b is gone from the source -> its chunks are removed,
    # doc_a's chunks remain.
    captured.clear()
    _patch_documents(monkeypatch, [doc_a])
    Indexer(config=_make_config()).run()
    assert _stored_urls(collection) == {doc_a.url}


def test_indexer_passes_configured_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured embedding model is forwarded to embed_texts."""
    from caracal_rag.indexing import Indexer

    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails.",
        content_hash=Document.hash("# Login\n\nDetails."),
    )
    config = _make_config()
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    _patch_documents(monkeypatch, [doc])
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    Indexer(config=config).run()
    assert captured.get("model") == config.embedding.model


def test_transient_fetch_failure_preserves_existing_vectors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Regression: a transient fetch failure must not be treated as deletion.

    Index two documents successfully, then on the next run load one document
    successfully and fail to fetch the other. The failed document is never
    yielded, so its URL is absent from ``current_urls``; the indexer must still
    preserve its existing vectors and skip destructive reconciliation entirely.
    """
    from caracal_rag.indexing import Indexer

    doc_a = Document(
        source="api",
        name="a.md",
        url="https://example.com/a.md",
        type="markdown",
        content="# A\n\nDetails here.",
        content_hash=Document.hash("# A\n\nDetails here."),
    )
    doc_b = Document(
        source="api",
        name="b.md",
        url="https://example.com/b.md",
        type="markdown",
        content="# B\n\nDetails here.",
        content_hash=Document.hash("# B\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    # First run: both documents are fetched and indexed successfully.
    _patch_documents(monkeypatch, [doc_a, doc_b])
    Indexer(config=_make_config()).run()
    assert _stored_urls(collection) == {doc_a.url, doc_b.url}

    # Second run: doc_a loads successfully but fetching doc_b fails
    # partway through. The failure carries only the already-loaded doc_a, so
    # doc_b's URL is not in current_urls -- if reconciliation ran it would be
    # treated as deleted.
    captured.clear()

    def _failing_load(path):
        yield doc_a
        raise SourceLoadError(
            "Failed to fetch https://example.com/b.md: connection reset",
            [doc_a],
        )

    monkeypatch.setattr(
        "caracal_rag.indexing.load_documents_from_config", _failing_load
    )

    Indexer(config=_make_config()).run()
    # Both documents' vectors remain: the failed doc_b was not deleted.
    assert _stored_urls(collection) == {doc_a.url, doc_b.url}
    # No re-embedding happened: the successfully loaded doc_a was unchanged.
    assert "texts" not in captured
    # Destructive reconciliation was skipped, not merely scoped to current_urls.
    assert "skipping deletion reconciliation" in capsys.readouterr().out


def test_changed_chunk_count_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document with stale extra chunks must have them removed.

    Detecting change only by content hash (or by iterating the new chunks)
    leaves stale vectors behind when the stored chunk-id set no longer matches
    the expected set. The indexer must compare the stored chunk-id set against
    the expected set and refresh the document.
    """
    from caracal_rag.indexing import Indexer

    text = "0123456789abcdefghij0123456789"
    document = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content=text,
        content_hash=Document.hash(text),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    # Patch chunk_document to yield 3 chunks for this document.
    monkeypatch.setattr(
        "caracal_rag.indexing.chunk_document",
        lambda doc, max_chars=1800: [
            chunking.Chunk(
                source=doc.source,
                document=doc.name,
                url=doc.url,
                type=doc.type,
                content_hash=doc.content_hash,
                chunk_index=i,
                text=t,
            )
            for i, t in enumerate(
                [doc.content[i : i + 10] for i in range(0, len(doc.content), 10)]
            )
        ],
    )

    _patch_documents(monkeypatch, [document])
    Indexer(config=_make_config()).run()
    assert len(collection._store) == 3  # 3 chunks stored initially

    # Simulate a stale extra chunk that was left behind by a previous buggy
    # indexing pass: same url/content_hash, but a chunk_index that no longer
    # exists in the current chunk layout.
    captured.clear()
    stale_chunk = chunking.Chunk(
        source=document.source,
        document=document.name,
        url=document.url,
        type=document.type,
        content_hash=document.content_hash,
        chunk_index=99,
        text="stale",
    )
    stale_metadata = {
        "source": document.source,
        "document": document.name,
        "url": document.url,
        "type": document.type,
        "content_hash": document.content_hash,
        "chunk_index": 99,
        "embedding_model": _make_config().embedding.model,
        "caracal_staging": False,
    }
    collection._store[_chunk_id(stale_chunk)] = {
        "text": stale_chunk.text,
        "embedding": [0.1, 0.2],
        "metadata": stale_metadata,
        **stale_metadata,
    }
    assert len(collection._store) == 4

    # Re-index the identical document: the stale chunk must be removed.
    Indexer(config=_make_config()).run()
    assert captured["texts"] != []  # changed doc was re-embedded
    assert len(collection._store) == 3
    assert all(
        meta.get("chunk_index") != 99 for meta in collection._store.values()
    )


def test_empty_document_removes_stale_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document that becomes empty must have its stale chunks removed.

    When ``chunk_document`` yields no chunks for a document that previously had
    vectors, the indexer must delete the stale chunks and skip embedding --
    not leave the old vectors in place.
    """
    from caracal_rag.indexing import Indexer

    original = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    _patch_documents(monkeypatch, [original])
    Indexer(config=_make_config()).run()
    assert len(collection._store) > 0

    # Next run: the document yields no chunks at all (e.g. it became empty).
    captured.clear()
    monkeypatch.setattr(
        "caracal_rag.indexing.chunk_document", lambda doc, **kwargs: []
    )
    _patch_documents(monkeypatch, [original])
    Indexer(config=_make_config()).run()
    assert len(collection._store) == 0  # stale chunks removed
    assert "texts" not in captured  # nothing was embedded


def test_whitespace_only_document_skips_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document whose content is whitespace-only must not upsert an empty chunk.

    ``chunk_markdown`` returns ``[""]`` for empty/whitespace content, so the
    indexer would otherwise upsert one empty chunk. Treat it as no content and
    remove any stale chunks instead.
    """
    from caracal_rag.indexing import Indexer

    original = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    _patch_documents(monkeypatch, [original])
    Indexer(config=_make_config()).run()
    assert len(collection._store) > 0

    # Next run: the document content is whitespace-only, so chunk_document
    # yields a single empty chunk.
    captured.clear()
    whitespace_doc = Document(
        source=original.source,
        name=original.name,
        url=original.url,
        type=original.type,
        content="   \n  \n",
        content_hash=Document.hash("   \n  \n"),
    )
    _patch_documents(monkeypatch, [whitespace_doc])
    Indexer(config=_make_config()).run()
    assert len(collection._store) == 0  # stale chunks removed, nothing embedded


def test_indexer_loads_collection_metadata_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 1: Collection metadata must be loaded once per run, not per document.

    The O(docs × collection) anti-pattern loads metadata for every document.
    The fixed indexer loads existing chunks once at __init__ and uses an
    in-memory cache thereafter.
    """
    from caracal_rag.indexing import Indexer, get_existing_chunks

    call_count = {"get_existing_chunks": 0}
    original_get = get_existing_chunks

    def counting_get(collection, **kwargs):
        call_count["get_existing_chunks"] += 1
        return original_get(collection, **kwargs)

    monkeypatch.setattr("caracal_rag.indexing.get_existing_chunks", counting_get)

    # Create 5 unchanged documents
    docs = [
        Document(
            source="api",
            name=f"doc{i}.md",
            url=f"https://example.com/doc{i}.md",
            type="markdown",
            content=f"Content {i}",
            content_hash=Document.hash(f"Content {i}"),
        )
        for i in range(5)
    ]
    _patch_documents(monkeypatch, docs)

    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    Indexer(config=_make_config()).run()

    # Should call get_existing_chunks exactly once (at __init__)
    assert call_count["get_existing_chunks"] == 1, (
        f"Expected 1 call to get_existing_chunks, got {call_count['get_existing_chunks']}"
    )


def test_embedding_failure_preserves_existing_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 2: Embedding failure must not delete existing vectors.

    The buggy flow deleted old vectors before embedding. The fixed flow
    embeds first, then deletes only on success.
    """
    from caracal_rag.indexing import Indexer

    # Seed collection with existing vectors
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    original_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    _patch_documents(monkeypatch, [original_doc])

    call_count = {"embed": 0}

    def fake_embed(texts, api_base, api_key=None, model=None):
        call_count["embed"] += 1
        if call_count["embed"] == 2:
            raise RuntimeError("Embedding service unavailable")
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("caracal_rag.indexing.embed_texts", fake_embed)

    # First run: index successfully
    Indexer(config=_make_config()).run()
    assert len(collection._store) > 0, "Should have indexed initial document"
    initial_store = dict(collection._store)

    # Second run: change document, but make embedding fail
    changed_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nUpdated details.",
        content_hash=Document.hash("# Login\n\nUpdated details."),
    )
    _patch_documents(monkeypatch, [changed_doc])

    try:
        Indexer(config=_make_config()).run()
    except RuntimeError as e:
        assert "Embedding service unavailable" in str(e)

    # Vectors should still be there (not deleted before embedding)
    assert collection._store == initial_store, (
        "Existing vectors were deleted despite embedding failure"
    )


def test_upsert_failure_preserves_existing_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacement must be safe against Chroma upsert failure too.

    The old chunks are no longer deleted before the new ones are written: the
    new chunks are upserted first (deterministic ids overwrite matching chunks in
    place) and only the obsolete ids are deleted afterwards. If the upsert itself
    fails, the working vectors must remain untouched and the in-memory cache
    must not have been modified, so the next run still sees the previous version.
    """
    from caracal_rag.indexing import Indexer

    # Seed collection with existing vectors
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    original_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    _patch_documents(monkeypatch, [original_doc])
    _patch_embed(monkeypatch, {})

    Indexer(config=_make_config()).run()
    assert len(collection._store) > 0, "Should have indexed initial document"
    initial_store = dict(collection._store)

    # Change the document and make the Chroma upsert fail.
    changed_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nUpdated details.",
        content_hash=Document.hash("# Login\n\nUpdated details."),
    )
    _patch_documents(monkeypatch, [changed_doc])

    def failing_upsert(*args, **kwargs):
        raise RuntimeError("Chroma upsert failed: connection reset")

    monkeypatch.setattr(FakeCollection, "upsert", failing_upsert)

    try:
        Indexer(config=_make_config()).run()
    except RuntimeError as exc:
        assert "Chroma upsert failed" in str(exc)

    # The old working vectors must be intact: nothing was deleted and the failed
    # upsert wrote nothing.
    assert collection._store == initial_store, (
        "Existing vectors were lost despite upsert failure"
    )


def test_legacy_collection_without_model_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection holding vectors with no embedding_model metadata must be refused.

    A legacy collection (created before model tracking existed, or populated
    outside the indexer) has an unknown embedding space. Indexing or searching
    against it would silently mix spaces, so the consistency guard must raise
    with actionable rebuild guidance instead of proceeding.
    """
    from caracal_rag.indexing import Indexer
    from caracal_rag.vectorstore import assert_embedding_model_consistent

    class LegacyCollection:
        def get(self, where=None, include=None):
            return {
                "ids": ["chunk-1"],
                "metadatas": [{"url": "https://example.com/old.md", "content_hash": "abc"}],
            }

        def query(self, query_embeddings, n_results: int = 5):
            return {"documents": [], "metadatas": [], "ids": [], "distances": []}

    config = _make_config()
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        assert_embedding_model_consistent(LegacyCollection(), config.embedding.model)

    # The indexer must refuse to run against a legacy collection too: the guard
    # runs at the start of run(), before any source loading or indexing.
    monkeypatch.setattr(
        "caracal_rag.indexing._chroma_client",
        lambda *args, **kwargs: type(
            "_C", (), {"get_or_create_collection": lambda self, name: LegacyCollection()}
        )(),
    )
    _patch_documents(monkeypatch, [])
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        Indexer(config=config).run()


def test_empty_collection_is_consistent_and_first_run_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty collection is consistent and the first run establishes the model.

    The invariant is collection-level, not per-vector: an empty collection has no
    vectors yet, so there is nothing to mix. The first successful indexing run
    then tags every stored vector with the configured model.
    """
    from caracal_rag.vectorstore import assert_embedding_model_consistent

    class EmptyCollection:
        def get(self, where=None, include=None):
            return {"ids": [], "metadatas": []}

    config = _make_config()
    # No exception -> the empty collection is consistent.
    assert_embedding_model_consistent(EmptyCollection(), config.embedding.model)


def test_embedding_model_change_detected_before_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3: Model mismatch must be detected before indexing.

    If stored vectors were created with a different embedding model,
    the indexer must raise ValueError with rebuild guidance.
    """
    from caracal_rag.indexing import Indexer

# Seed collection with vectors from a different model
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    # Use the real upsert path so the store has the right shape
    seed_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login",
        content_hash=Document.hash("# Login"),
    )
    _patch_documents(monkeypatch, [seed_doc])
    Indexer(config=_make_config()).run()
    # Now overwrite the stored embedding model to simulate a different model
    for record in collection._store.values():
        record["embedding_model"] = "text-embedding-3-large"
        record["metadata"]["embedding_model"] = "text-embedding-3-large"

    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login",
        content_hash=Document.hash("# Login"),
    )
    _patch_documents(monkeypatch, [doc])

    # Config still uses text-embedding-3-small (different from stored)
    config = _make_config()

    with pytest.raises(ValueError, match="Mixing embedding spaces"):
        Indexer(config=config).run()


def test_stored_vectors_carry_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3b: Upserted vectors must store the embedding model in metadata."""
    from caracal_rag.indexing import Indexer

    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)

    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login",
        content_hash=Document.hash("# Login"),
    )
    _patch_documents(monkeypatch, [doc])

    config = _make_config()
    expected_model = config.embedding.model

    Indexer(config=config).run()

    # Check that stored metadata includes embedding_model
    assert len(collection._store) > 0
    for record in collection._store.values():
        meta = record.get("metadata") or {}
        assert meta.get("embedding_model") == expected_model, (
            f"Expected embedding_model={expected_model}, got {meta.get('embedding_model')}"
        )


def test_mixed_collection_with_untagged_vectors_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection holding one tagged and one untagged vector must be refused.

    ``collection_is_legacy`` previously accepted a non-empty collection as long
    as at least one vector carried ``embedding_model``, so vectors from an
    unknown embedding space could be mixed with query vectors from the
    configured model during both indexing and search. The whole collection is
    unsafe when any vector lacks model metadata, so the guard must raise.
    """
    from caracal_rag.indexing import Indexer
    from caracal_rag.vectorstore import assert_embedding_model_consistent

    class MixedCollection:
        def get(self, where=None, include=None):
            return {
                "ids": ["chunk-tagged", "chunk-untagged"],
                "metadatas": [
                    {"embedding_model": "text-embedding-3-small"},
                    {"url": "https://example.com/unknown.md"},
                ],
            }

        def query(self, query_embeddings, n_results: int = 5):
            return {"documents": [], "metadatas": [], "ids": [], "distances": []}

    config = _make_config()
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        assert_embedding_model_consistent(MixedCollection(), config.embedding.model)

    # The indexer must refuse to run against a mixed collection too: the guard
    # runs at the start of run(), before any source loading or indexing.
    monkeypatch.setattr(
        "caracal_rag.indexing._chroma_client",
        lambda *args, **kwargs: type(
            "_C", (), {"get_or_create_collection": lambda self, name: MixedCollection()}
        )(),
    )
    _patch_documents(monkeypatch, [])
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        Indexer(config=config).run()


def test_partial_upsert_failure_preserves_existing_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Chroma upsert that mutates existing ids before failing must not lose data.

    The previous test only simulated an exception before any write. The fixed
    indexer stages the replacement under temporary ids, so a partial failure can
    never overwrite the old working vectors in place: the old deterministic ids
    are untouched until the staging write has fully succeeded, and a partial
    staging write is rolled back. This test makes the upsert write at least one
    existing id and then raise, and asserts the old vectors remain intact.
    """
    from caracal_rag.indexing import Indexer

    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    original_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    _patch_documents(monkeypatch, [original_doc])
    _patch_embed(monkeypatch, {})

    Indexer(config=_make_config()).run()
    assert len(collection._store) > 0
    initial_store = dict(collection._store)

    # Change the document and make the Chroma upsert fail *after* writing at
    # least one existing id, so the failure is ambiguous/partial rather than a
    # clean pre-write exception.
    changed_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nUpdated details.",
        content_hash=Document.hash("# Login\n\nUpdated details."),
    )
    _patch_documents(monkeypatch, [changed_doc])

    written_existing: list[str] = []

    def failing_upsert(self, ids, documents, embeddings, metadatas):
        # Deterministic ids overwrite existing vectors in place before the
        # failure, simulating a partial Chroma upsert.
        for cid, text, meta in zip(ids, documents, metadatas):
            if cid in self._store:
                written_existing.append(cid)
            self._store[cid] = {
                "text": text,
                "embedding": embeddings[0] if embeddings else None,
                "metadata": meta,
                **meta,
            }
        raise RuntimeError("Chroma upsert failed: connection reset")

    monkeypatch.setattr(FakeCollection, "upsert", failing_upsert)

    try:
        Indexer(config=_make_config()).run()
    except RuntimeError as exc:
        assert "Chroma upsert failed" in str(exc)

    # The old working vectors must be intact: nothing was deleted and the failed
    # upsert wrote nothing that survives the rollback.
    assert collection._store == initial_store, (
        "Existing vectors were lost despite upsert failure"
    )
    assert written_existing == []  # the old deterministic ids were never touched


def test_untagged_document_chunk_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for discussion_r4282720190: a mixed collection is refused.

    Previously a collection holding one tagged and one untagged vector was
    accepted at the collection level, and ``_is_unchanged`` was expected to
    detect the per-document drift and refresh that document. The safer
    behaviour is to refuse the mixed state outright: any vector that lacks
    ``embedding_model`` makes the whole collection unsafe, so the consistency
    guard raises with rebuild guidance before indexing or searching.
    """
    from caracal_rag.indexing import Indexer

    current_model = _make_config().embedding.model
    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)

    # Index two documents so the collection is tagged with the current model.
    other_doc = Document(
        source="api",
        name="other.md",
        url="https://example.com/other.md",
        type="markdown",
        content="# Other\n\nUnrelated.",
        content_hash=Document.hash("# Other\n\nUnrelated."),
    )
    _patch_documents(monkeypatch, [doc, other_doc])
    Indexer(config=_make_config()).run()
    assert _stored_models(collection) == {current_model}

    # Simulate per-document drift: strip the model metadata from one document's
    # stored chunks while leaving the other tagged. The collection is now mixed.
    for record in collection._store.values():
        if record.get("url") == doc.url:
            record.pop("embedding_model", None)
            record.get("metadata", {}).pop("embedding_model", None)
    assert _stored_models(collection) == {current_model}

    # Re-indexing the identical document must refuse the mixed collection
    # rather than silently mixing the untagged vectors with new query vectors.
    captured.clear()
    _patch_documents(monkeypatch, [doc])
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        Indexer(config=_make_config()).run()
    assert captured.get("model") != current_model or "texts" not in captured


def test_failed_promotion_hides_staging_and_retry_reconciles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed promotion must not leak temporary rows; retry must fully reconcile."""
    from caracal_rag.indexing import Indexer

    config = _make_config()
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    captured: dict[str, Any] = {}
    _patch_embed(monkeypatch, captured)

    old_doc = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Login\n\nOld content."
    )
    new_doc = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Login\n\nNew content."
    )
    _patch_documents(monkeypatch, [old_doc])
    Indexer(config=config).run()
    assert collection._store

    original_upsert = FakeCollection.upsert
    promotion_attempts = 0

    def fail_promotion_after_partial_write(self, ids, documents, embeddings, metadatas):
        nonlocal promotion_attempts
        if all(meta.get("caracal_staging", False) for meta in metadatas):
            return original_upsert(self, ids, documents, embeddings, metadatas)
        # Rollback writes the old content hash and must be allowed to succeed.
        if all(meta.get("content_hash") == old_doc.content_hash for meta in metadatas):
            return original_upsert(self, ids, documents, embeddings, metadatas)
        promotion_attempts += 1
        # Mutate the deterministic row before raising to model an ambiguous
        # remote failure after a partial/committed write.
        original_upsert(self, ids[:1], documents[:1], embeddings[:1], metadatas[:1])
        raise RuntimeError("simulated promotion failure after partial write")

    monkeypatch.setattr(FakeCollection, "upsert", fail_promotion_after_partial_write)
    _patch_documents(monkeypatch, [new_doc])
    with pytest.raises(RuntimeError, match="Chroma upsert failed"):
        Indexer(config=config).run()

    assert promotion_attempts == 2
    expected_ids = {_chunk_id(chunk) for chunk in chunking.chunk_document(new_doc)}
    assert expected_ids.issubset(collection._store)
    assert any(
        record.get("metadata", {}).get("caracal_staging", False)
        for record in collection._store.values()
    )
    assert_embedding_model_consistent(collection, config.embedding.model)

    # Force a staged row into top-k. Search filters it and returns the last
    # committed version, rather than exposing staging or a partial replacement.
    result = search_chunks(collection, [0.1, 0.2], top_k=1)
    assert result["ids"][0][0] in expected_ids
    assert len(result["metadatas"][0]) == 1
    assert result["metadatas"][0][0].get("caracal_staging", False) is False
    assert "Old content." in result["documents"][0][0]
    assert "New content." not in result["documents"][0][0]

    # A retry discovers prior staging rows, refreshes the document successfully,
    # and leaves exactly the deterministic new-content ID set.
    monkeypatch.setattr(FakeCollection, "upsert", original_upsert)
    Indexer(config=config).run()
    assert set(collection._store) == expected_ids
    assert all(
        record.get("metadata", {}).get("caracal_staging", False) is False
        for record in collection._store.values()
    )



def test_listing_failure_never_reconciles_or_reports_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed inventory keeps all old rows and marks the run incomplete."""
    from caracal_rag.indexing import Indexer

    existing = _make_markdown_doc(
        "api", "old.md", "https://example.com/old.md", "# Old content"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [existing])
    Indexer(config=_make_config()).run()
    before = {key: dict(value) for key, value in collection._store.items()}
    assert before

    def failed_inventory(_path):
        raise SourceLoadError("GitHub inventory request failed", [])

    monkeypatch.setattr("caracal_rag.indexing.load_documents_from_config", failed_inventory)
    result = Indexer(config=_make_config()).run()

    assert result is False
    assert collection._store == before


def test_changed_document_embedding_failure_does_not_delete_stale_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale cleanup must happen after embeddings for all discovered docs succeed."""
    from caracal_rag.indexing import Indexer

    original = _make_markdown_doc(
        "api", "current.md", "https://example.com/current.md", "# Current old version"
    )
    stale = _make_markdown_doc(
        "api", "removed.md", "https://example.com/removed.md", "# Still valid until reconcile"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [original, stale])
    Indexer(config=_make_config()).run()
    before = {key: dict(value) for key, value in collection._store.items()}
    assert _stored_urls(collection) == {original.url, stale.url}

    changed = _make_markdown_doc(
        "api", "current.md", original.url, "# Current changed version"
    )
    _patch_documents(monkeypatch, [changed])

    def failing_embed(*args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr("caracal_rag.indexing.embed_texts", failing_embed)
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        Indexer(config=_make_config()).run()

    assert collection._store == before
    assert _stored_urls(collection) == {original.url, stale.url}


def test_unexpected_empty_source_refuses_destructive_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty inventory cannot wipe an existing source unless opted in."""
    from caracal_rag.indexing import Indexer

    existing = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Login"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [existing])
    Indexer(config=_make_config()).run()
    before = {key: dict(value) for key, value in collection._store.items()}

    _patch_documents(monkeypatch, [])
    with pytest.raises(RuntimeError, match="resolved to zero documents"):
        Indexer(config=_make_config()).run()

    assert collection._store == before


def test_allow_empty_source_is_an_explicit_delete_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_empty=true intentionally permits a complete empty inventory to clear a source."""
    from caracal_rag.indexing import Indexer

    existing = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Login"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [existing])
    Indexer(config=_make_config()).run()
    assert collection._store

    config = _make_config()
    config.sources[0].allow_empty = True
    _patch_documents(monkeypatch, [])
    assert Indexer(config=config).run() is True

    assert collection._store == {}


def test_empty_document_removal_is_deferred_until_all_embeddings_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank document is deleted only after other current documents index successfully."""
    from caracal_rag.indexing import Indexer

    original_blank = _make_markdown_doc(
        "api", "blank.md", "https://example.com/blank.md", "# Previously valid"
    )
    original_other = _make_markdown_doc(
        "api", "other.md", "https://example.com/other.md", "# Other previous version"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [original_blank, original_other])
    Indexer(config=_make_config()).run()
    before = {key: dict(value) for key, value in collection._store.items()}

    blank_now = _make_markdown_doc(
        "api", "blank.md", original_blank.url, "   \n\n  "
    )
    other_now = _make_markdown_doc(
        "api", "other.md", original_other.url, "# Other changed version"
    )
    _patch_documents(monkeypatch, [blank_now, other_now])

    def failing_embed(*args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr("caracal_rag.indexing.embed_texts", failing_embed)
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        Indexer(config=_make_config()).run()

    assert collection._store == before


def test_partial_stale_delete_failure_is_reported_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially applied stale delete is surfaced and leaves current rows intact."""
    from caracal_rag.indexing import Indexer

    current = _make_markdown_doc(
        "api", "current.md", "https://example.com/current.md", "# Current"
    )
    stale = _make_markdown_doc(
        "api", "stale.md", "https://example.com/stale.md", "# Stale"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [current, stale])
    Indexer(config=_make_config()).run()
    _patch_documents(monkeypatch, [current])

    def partial_delete(collection_arg, current_urls, source=None):
        stale_ids = [
            cid for cid, record in collection_arg._store.items()
            if record.get("url") == stale.url
        ]
        collection_arg.delete(ids=stale_ids[:1])
        raise RuntimeError("connection reset after delete")

    monkeypatch.setattr("caracal_rag.indexing.delete_missing_documents", partial_delete)
    with pytest.raises(RuntimeError, match="manifest was not updated"):
        Indexer(config=_make_config()).run()

    assert current.url in _stored_urls(collection)


def test_failed_staging_cleanup_is_reported_and_staging_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rollback is explicit, and any orphan staging rows remain unsearchable."""
    from caracal_rag.indexing import Indexer
    from caracal_rag.vectorstore import search_chunks

    original_doc = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Previous committed content"
    )
    collection = FakeCollection()
    _patch_client(monkeypatch, collection)
    _patch_embed(monkeypatch, {})
    _patch_documents(monkeypatch, [original_doc])
    Indexer(config=_make_config()).run()
    old_store = {key: dict(value) for key, value in collection._store.items()}

    changed = _make_markdown_doc(
        "api", "login.md", original_doc.url, "# Replacement content"
    )
    _patch_documents(monkeypatch, [changed])
    from caracal_rag.vectorstore import upsert_chunks as real_upsert_chunks
    original_delete_method = FakeCollection.delete

    def partial_stage_then_fail(collection_arg, chunks, embeddings, embedding_model,
                                ids=None, staging=False, embedding_dim=None):
        chunk_list = list(chunks)
        if staging:
            real_upsert_chunks(
                collection_arg,
                chunk_list[:1],
                embeddings[:1],
                embedding_model=embedding_model,
                ids=list(ids or [])[:1],
                staging=True,
                embedding_dim=embedding_dim,
            )
            raise RuntimeError("injected partial stage failure")
        return real_upsert_chunks(
            collection_arg, chunk_list, embeddings, embedding_model=embedding_model,
            ids=ids, staging=staging, embedding_dim=embedding_dim
        )

    def fail_stage_cleanup(self, ids):
        if any(
            (self._store.get(cid, {}).get("metadata", {}).get("caracal_staging", False))
            for cid in ids
        ):
            raise RuntimeError("injected cleanup failure")
        return original_delete_method(self, ids)

    monkeypatch.setattr("caracal_rag.indexing.upsert_chunks", partial_stage_then_fail)
    monkeypatch.setattr(FakeCollection, "delete", fail_stage_cleanup)

    with pytest.raises(RuntimeError, match="cleanup of temporary rows also failed"):
        Indexer(config=_make_config()).run()

    assert all(collection._store[cid] == record for cid, record in old_store.items())
    staging_ids = [
        cid for cid, record in collection._store.items()
        if record.get("metadata", {}).get("caracal_staging", False)
    ]
    assert staging_ids
    results = search_chunks(collection, [0.1, 0.2], top_k=5)
    returned_metadata = results.get("metadatas", [[]])[0]
    assert all(not (meta or {}).get("caracal_staging", False) for meta in returned_metadata)


def test_indexer_handles_numpy_embedding_arrays_from_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chroma returns embeddings as NumPy arrays; their truth value is undefined."""
    import numpy as np

    from caracal_rag.indexing import Indexer

    class NumpyEmbeddingsCollection(FakeCollection):
        def get(self, where=None, include=None):
            result = super().get(where=where, include=include)
            if include is not None and "embeddings" in include:
                result["embeddings"] = np.asarray(
                    result.get("embeddings", []), dtype=np.float32
                )
            return result

    original = _make_markdown_doc(
        "api", "login.md", "https://example.com/login.md", "# Login\n\nOld details."
    )
    updated = _make_markdown_doc(
        "api", "login.md", original.url, "# Login\n\nUpdated details."
    )
    _patch_embed(monkeypatch, {})
    collection = NumpyEmbeddingsCollection()
    _patch_client(monkeypatch, collection)
    _patch_documents(monkeypatch, [original])

    Indexer(config=_make_config()).run()
    assert _stored_urls(collection) == {original.url}

    # Updating a previously indexed document snapshots existing vectors for
    # rollback; this is the path where ndarray truthiness used to crash.
    _patch_documents(monkeypatch, [updated])
    Indexer(config=_make_config()).run()
    assert _stored_urls(collection) == {updated.url}
    assert all(
        record.get("content_hash") == updated.content_hash
        for record in collection._store.values()
    )

