from __future__ import annotations

import uuid
from typing import Iterable

from caracal_rag.chunking import Chunk, chunk_document
from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.sources import (
    Document,
    SourceLoadError,
    load_documents_from_config,
)
from caracal_rag.vectorstore import (
    _chroma_client,
    _chunk_id,
    assert_embedding_model_consistent,
    delete_document_chunks,
    delete_missing_documents,
    ensure_collection,
    get_existing_chunks,
    upsert_chunks,
)


class Indexer:
    def __init__(self, config: AppConfig, source_filter: str | None = None) -> None:
        self.config = config
        self.source_filter = source_filter
        self.embedding_model = config.embedding.model
        self.client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        self.collection = ensure_collection(self.client, config.chroma.collection)
        # Load the collection metadata once per run instead of once per document.
        # Fetching it inside _index_document made the cost O(document_count x
        # collection_size); the cache is refreshed after every upsert/delete so
        # the next document still sees an up-to-date view.
        self._existing_chunks = get_existing_chunks(self.collection)

    def _load_documents(self) -> Iterable[Document]:
        for doc in load_documents_from_config("config/sources.example.yaml"):
            if self.source_filter and doc.source != self.source_filter:
                continue
            yield doc

    def run(self) -> None:
        # Fail fast if the collection already holds vectors from a different
        # embedding model. Mixing embedding spaces produces meaningless search
        # results, so refuse to index rather than silently mixing them.
        assert_embedding_model_consistent(self.collection, self.embedding_model)
        try:
            documents = list(self._load_documents())
        except SourceLoadError as exc:
            # Source loading partially failed: some documents were fetched but
            # one failed transiently. Index what we have and skip destructive
            # reconciliation entirely, so a transient fetch failure is never
            # mistaken for document deletion.
            documents = exc.documents
            if self.source_filter:
                documents = [
                    doc for doc in documents if doc.source == self.source_filter
                ]
            print(
                "Source loading partially failed; skipping deletion reconciliation "
                f"({len(documents)} document(s) loaded): {exc}"
            )
        else:
            current_urls = {doc.url for doc in documents}

            # Remove chunks for documents that no longer exist in the source.
            # Reconciliation is scoped to the source being indexed so that running
            # `index --source fiteo-docs` never deletes vectors from other sources.
            stale = delete_missing_documents(
                self.collection, current_urls, source=self.source_filter
            )
            if stale:
                print(f"Removed stale chunks for {len(stale)} deleted document(s)")

        for document in documents:
            self._index_document(document)

    def _index_document(self, document: Document) -> None:
        # Skip unchanged documents: if every chunk already exists with the same
        # content hash, there is nothing to embed or upsert. The lookup uses the
        # in-memory chunk metadata loaded once per run (see __init__), so this is
        # O(chunks in the collection) at construction, not once per document.
        if self._is_unchanged(document, self._existing_chunks):
            return

        chunks = list(chunk_document(document))
        # A document whose content is whitespace-only still yields one empty
        # chunk via chunk_markdown; treat that as no content so stale chunks are
        # removed instead of an empty chunk being upserted.
        if not chunks or all(not chunk.text.strip() for chunk in chunks):
            delete_document_chunks(self.collection, document.url)
            self._drop_cached_chunks(document.url)
            return

        # Generate the replacement embeddings BEFORE touching the working vectors
        # for the current document. If embedding generation fails (LiteLLM down,
        # rate limit, bad model), the valid old version is preserved instead of
        # being silently lost.
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(
            texts,
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
            model=self.embedding_model,
        )

        final_ids = [_chunk_id_for(chunk) for chunk in chunks]
        # Stage the replacement under temporary ids. Staging ids are random uuid4
        # values that do not collide with any existing chunk id, so a partial
        # Chroma upsert can never overwrite the old working vectors in place: the
        # old document's deterministic ids are untouched until the staging write
        # has fully succeeded. If the staging upsert fails, the old vectors and the
        # in-memory cache are left exactly as they were.
        staging_ids = [str(uuid.uuid4()) for _ in chunks]
        try:
            upsert_chunks(
                self.collection,
                chunks,
                embeddings,
                embedding_model=self.embedding_model,
                ids=staging_ids,
                staging=True,
            )
        except Exception as exc:
            # Roll back the staged ids a partial failure may have written: they
            # do not belong to any document and would otherwise linger in the
            # collection. The old working vectors were never touched, so after
            # the rollback the collection is exactly as it was before the run.
            try:
                self.collection.delete(ids=staging_ids)
            except Exception:
                pass  # best effort: a lingering staged id is detected as drift
            raise exc

        # The new content is safely in the collection. Remove only the old
        # document chunks known to the in-memory cache -- NOT the staged copies,
        # which share the document url and would otherwise be removed too -- so
        # the new content survives even if the promotion step below fails.
        old_ids = [
            chunk_id
            for chunk_id, metadata in self._existing_chunks.items()
            if (metadata or {}).get("url") == document.url
        ]
        if old_ids:
            self.collection.delete(ids=old_ids)
            for chunk_id in old_ids:
                self._existing_chunks.pop(chunk_id, None)

        # Promote the staged chunks to their deterministic ids. These ids are now
        # free (the old chunks were removed), so the write overwrites nothing. If
        # it fails partway, some final ids may already hold the new content; the
        # staged copies are still in the collection, so retrying the promotion is
        # safe and idempotent -- it re-writes every final id with identical
        # content.
        try:
            upsert_chunks(
                self.collection,
                chunks,
                embeddings,
                embedding_model=self.embedding_model,
                ids=final_ids,
            )
        except Exception:
            try:
                upsert_chunks(
                    self.collection,
                    chunks,
                    embeddings,
                    embedding_model=self.embedding_model,
                    ids=final_ids,
                )
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Chroma upsert failed and the retry failed too. The document's "
                    "old vectors were already removed but the replacement is still "
                    "present under temporary staging ids; re-run indexing to "
                    "reconcile the collection."
                ) from rollback_exc

        # Promotion succeeded: the temporary staging ids are now redundant.
        self.collection.delete(ids=staging_ids)
        self._upsert_cached_chunks(chunks)

    def _drop_cached_chunks(self, document_url: str) -> None:
        """Remove the document's chunks from the in-memory metadata cache."""
        self._existing_chunks = {
            chunk_id: metadata
            for chunk_id, metadata in self._existing_chunks.items()
            if (metadata or {}).get("url") != document_url
        }

    def _upsert_cached_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Reflect an upsert in the in-memory metadata cache.

        The metadata written to the collection is the same the indexer computed,
        so it can be mirrored in memory without another collection read. This
        keeps the per-document work bounded and the cache accurate for the
        remaining documents in the run.
        """
        for chunk in chunks:
            self._existing_chunks[_chunk_id(chunk)] = {
                "source": chunk.source,
                "document": chunk.document,
                "url": chunk.url,
                "type": chunk.type,
                "content_hash": chunk.content_hash,
                "chunk_index": chunk.chunk_index,
                "embedding_model": self.embedding_model,
                "caracal_staging": False,
            }

    def _is_unchanged(self, document: Document, existing: dict[str, dict]) -> bool:
        """Return True when every chunk for this document already exists and matches.

        Compares the expected chunk-id set against the ids actually stored for
        this document, so a shrinking chunk count (or a document that became
        empty) is detected as a change and the document is refreshed. Checking
        only the new chunks would leave stale vectors behind when the chunk
        layout changes while the text stays the same.

        The embedding model that produced the stored vectors is also compared:
        if the configured model changed, the old vectors live in a different
        embedding space and must never be treated as current.
        """
        expected_ids = {_chunk_id_for(chunk) for chunk in chunk_document(document)}
        stored_ids = {
            cid
            for cid, metadata in existing.items()
            if (metadata or {}).get("url") == document.url
        }
        if stored_ids != expected_ids:
            return False
        for chunk in chunk_document(document):
            chunk_id = _chunk_id_for(chunk)
            metadata = existing.get(chunk_id)
            if metadata is None:
                return False
            if metadata.get("content_hash") != document.content_hash:
                return False
            if metadata.get("url") != chunk.url or metadata.get("chunk_index") != chunk.chunk_index:
                return False
            if metadata.get("embedding_model") != self.embedding_model:
                return False
        return True


def _chunk_id_for(chunk: Chunk) -> str:
    return _chunk_id(chunk)
