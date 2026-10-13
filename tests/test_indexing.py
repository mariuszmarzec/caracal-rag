"""Tests for the indexing pipeline: idempotency, change, and deletion behaviour.

External services (LiteLLM, GitHub, Chroma) are replaced with in-memory fakes so
the tests run without network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from caracal_rag.config import AppConfig, SourceConfig
from caracal_rag.sources import Document


class FakeCollection:
    """Minimal in-memory Chroma collection."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(
        self,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        if where is None:
            return {
                "ids": list(self._store.keys()),
                "metadatas": [self._store[cid] for cid in self._store],
            }
        url = where.get("url")
        matches = {
            cid: meta
            for cid, meta in self._store.items()
            if meta.get("url") == url
        }
        return {
            "ids": list(matches.keys()),
            "metadatas": list(matches.values()),
        }

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
    def fake_embed(texts, api_base, api_key=None):
        captured["texts"] = list(texts)
        captured["api_base"] = api_base
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


def test_indexer_skips_empty_documents(monkeypatch: pytest.MonkeyPatch) -> None:
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
