"""Tests for the indexing pipeline: idempotency, change, and deletion behaviour.

External services (LiteLLM, GitHub, Chroma) are replaced with in-memory fakes so
the tests run without network access.

The :class:`FakeCollection` faithfully honours the Chroma ``include`` argument:
when ``metadatas`` is not requested it is omitted from the result, mirroring the
real Chroma behaviour that motivated the ``delete_missing_documents`` fix.
"""

from __future__ import annotations

from typing import Any

import pytest

from caracal_rag.config import AppConfig, SourceConfig
from caracal_rag.sources import Document, SourceLoadError


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
        if include is not None and "metadatas" in include:
            result["metadatas"] = [meta for _, meta in items]
        return result

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for cid, text, meta in zip(ids, documents, metadatas):
            self._store[cid] = {
                "text": text,
                "embedding": embeddings[0] if embeddings else None,
                "metadata": meta,
                **meta,
            }

    def add(self, **kwargs) -> None:
        self.upsert(**kwargs)

    def delete(self, ids) -> None:
        for cid in ids:
            self._store.pop(cid, None)

    def query(self, query_embeddings, n_results: int = 5) -> dict[str, Any]:
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
        }


def _make_config() -> AppConfig:
    return AppConfig(
        sources=[SourceConfig(name="api", type="markdown", urls=["https://example.com/login.md"])],
        embedding=AppConfig.from_yaml("config/sources.example.yaml").embedding,
        chroma=AppConfig.from_yaml("config/sources.example.yaml").chroma,
    )


def _patch_embed(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    def fake_embed(texts, api_base, api_key=None, model=None):
        captured["texts"] = list(texts)
        captured["api_base"] = api_base
        captured["model"] = model
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("caracal_rag.indexing.embed_texts", fake_embed)


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

    indexer = Indexer(config=_make_config())
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
