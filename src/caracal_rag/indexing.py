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
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    delete_document_chunks,
    delete_missing_documents,
    ensure_collection,
    get_existing_chunks,
    upsert_chunks,
)


class Indexer:
    def __init__(
        self,
        config: AppConfig,
        source_filter: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.config = config
        self.source_filter = source_filter
        self.embedding_model = config.embedding.model
        #: Dimensionality of the configured embedding model, probed before the
        #: run. ``None`` when the caller did not supply one; the first document
        #: actually embedded establishes it, and any later document that
        #: disagrees is refused so a mid-run dimension drift can never silently
        #: mix vectors from two embedding spaces.
        self.embedding_dim = embedding_dim
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
        self._pending_empty_deletions: set[str] = set()

    def _load_documents(self) -> Iterable[Document]:
        if self.config.config_path is None:
            raise ValueError(
                "Indexer requires a configuration with a resolved config_path. "
                "Load the active installed configuration via `load_active_config()` "
                "and pass it to the indexer."
            )
        for doc in load_documents_from_config(self.config.config_path):
            if self.source_filter and doc.source != self.source_filter:
                continue
            yield doc

    def run(self) -> bool:
        self._pending_empty_deletions.clear()
        # Fail fast if the collection already holds vectors from a different
        # embedding model. Mixing embedding spaces produces meaningless search
        # results, so refuse to index rather than silently mixing them.
        assert_embedding_model_consistent(self.collection, self.embedding_model)
        if self.embedding_dim is not None:
            assert_embedding_dim_consistent(self.collection, self.embedding_dim)

        configured_sources = list(self.config.sources or [])
        relevant_sources = [
            source
            for source in configured_sources
            if self.source_filter is None or source.name == self.source_filter
        ]

        partial_load = False
        try:
            documents = list(self._load_documents())
        except SourceLoadError as exc:
            # A partial or failed inventory is never authoritative enough to
            # delete anything. Index successfully fetched documents, but defer
            # all reconciliation until a later run obtains a complete inventory.
            documents = exc.documents
            if self.source_filter:
                documents = [
                    doc for doc in documents if doc.source == self.source_filter
                ]
            partial_load = True
            print(
                "Source loading partially failed; skipping deletion reconciliation "
                f"({len(documents)} document(s) loaded): {exc}"
            )

        if not partial_load:
            loaded_sources = {document.source for document in documents}
            existing_sources = {
                (metadata or {}).get("source")
                for metadata in self._existing_chunks.values()
                if not (metadata or {}).get("caracal_staging", False)
            }
            unexpectedly_empty = [
                source.name
                for source in relevant_sources
                if source.name in existing_sources
                and source.name not in loaded_sources
                and not source.allow_empty
            ]
            if not relevant_sources and existing_sources and self.source_filter is None:
                raise RuntimeError(
                    "Indexing aborted because no sources are configured while committed "
                    "Chroma chunks still exist. No documents were indexed or deleted. "
                    "Restore the intended sources.yaml before indexing."
                )
            if unexpectedly_empty:
                names = ", ".join(sorted(unexpectedly_empty))
                raise RuntimeError(
                    "Indexing aborted because configured source(s) resolved to zero "
                    f"documents while committed chunks still exist: {names}. No "
                    "documents were indexed or deleted. Check the source path, URL, "
                    "repository and branch. If an empty inventory is intentional and "
                    "should clear that source, set allow_empty: true for that source "
                    "in sources.yaml and rerun indexing."
                )

        # Index/refresh every discovered document before deleting stale rows.
        # If any embedding or Chroma write fails, reconciliation is skipped, so
        # the failure cannot additionally remove still-valid documents that were
        # absent from the current inventory. Per-document replacement uses staged
        # rows and rollback; the run as a whole is intentionally not transactional.
        for document in documents:
            self._index_document(document)

        if not partial_load:
            current_urls = {doc.url for doc in documents}
            try:
                # Empty-content documents are explicit inventory entries, so defer
                # their deletion until every other document has indexed successfully.
                for document_url in sorted(self._pending_empty_deletions):
                    delete_document_chunks(self.collection, document_url)
                    self._drop_cached_chunks(document_url)

                # This is safe only after a complete inventory AND successful
                # indexing of every discovered document. A partial Chroma delete
                # can leave stale rows behind; that is non-destructive and the
                # next successful run retries reconciliation.
                stale = delete_missing_documents(
                    self.collection, current_urls, source=self.source_filter
                )
            except Exception as exc:
                raise RuntimeError(
                    "Indexing updated discovered documents but failed while deleting "
                    "stale Chroma rows. The deletion may have been partially applied; "
                    "valid current rows are retained, and rerunning indexing after the "
                    "Chroma failure is resolved will reconcile remaining stale rows. "
                    "The manifest was not updated."
                ) from exc
            if stale:
                print(f"Removed stale chunks for {len(stale)} deleted document(s)")

        # False means useful documents may have been indexed, but the inventory
        # was incomplete and stale-row reconciliation was intentionally skipped.
        # The CLI must not update the manifest or report a complete successful run.
        return not partial_load

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
            # This document is explicitly present but no longer has content.
            # Defer deleting its old chunks until all other documents finish,
            # otherwise a later embedding failure would leave a partial run.
            self._pending_empty_deletions.add(document.url)
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
            input_type="passage",
        )

        # The dimension is established on the first successful embedding of the
        # run and validated on every later document. A mid-run dimension drift
        # (the configured model silently producing a different family) must
        # never be allowed to mix vectors from two embedding spaces, so refuse
        # rather than upsert the drifted document.
        observed_dim = len(embeddings[0]) if embeddings and embeddings[0] else None
        if observed_dim is not None:
            if self.embedding_dim is None:
                self.embedding_dim = observed_dim
            elif self.embedding_dim != observed_dim:
                raise RuntimeError(
                    f"Embedding dimension changed mid-run: the run started with "
                    f"{self.embedding_dim}-dimensional vectors but document "
                    f"{document.url!r} produced {observed_dim}-dimensional ones. "
                    "Mixing embedding spaces produces meaningless search results; "
                    "aborting the run."
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
                embedding_dim=self.embedding_dim,
            )
        except Exception as exc:
            # A failed staging upsert may have written only part of the batch.
            # Remove those temporary ids without hiding a failed cleanup: search
            # excludes rows marked caracal_staging, but an operator needs to know
            # when an orphaned stage row may remain for later reconciliation.
            try:
                self.collection.delete(ids=staging_ids)
            except Exception as cleanup_exc:
                try:
                    self._existing_chunks = get_existing_chunks(self.collection)
                except Exception:
                    pass
                raise RuntimeError(
                    "Chroma upsert failed during staging and cleanup of temporary rows also "
                    "failed. Any remaining rows are marked caracal_staging and excluded "
                    "from search. Investigate Chroma, then rerun indexing to reconcile."
                ) from cleanup_exc
            raise RuntimeError(
                "Chroma upsert failed during staging; partial temporary rows were removed and "
                "the previous committed document remains intact."
            ) from exc

        # Snapshot the previous committed version before replacing deterministic ids.
        # Chroma upsert may partially mutate a batch and then fail, so a metadata-only
        # cache is not enough to restore the last known-good vectors.
        previous = self.collection.get(
            where={"url": document.url},
            include=["documents", "embeddings", "metadatas"],
        )
        previous_ids = previous.get("ids") or []
        previous_documents = previous.get("documents") or []
        previous_embeddings = previous.get("embeddings")
        if previous_embeddings is None:
            previous_embeddings = []
        previous_metadatas = previous.get("metadatas") or []
        snapshot_ids: list[str] = []
        snapshot_documents: list[str] = []
        snapshot_embeddings: list[list[float]] = []
        snapshot_metadatas: list[dict] = []
        for index, chunk_id in enumerate(previous_ids):
            metadata = (previous_metadatas[index] if index < len(previous_metadatas) else {}) or {}
            if metadata.get("caracal_staging", False):
                continue
            if index >= len(previous_documents) or index >= len(previous_embeddings):
                raise RuntimeError(
                    "Cannot safely replace document because Chroma did not return its "
                    "previous documents and embeddings for rollback."
                )
            snapshot_ids.append(chunk_id)
            snapshot_documents.append(previous_documents[index])
            snapshot_embeddings.append(list(previous_embeddings[index]))
            snapshot_metadata = dict(metadata)
            snapshot_metadata["caracal_staging"] = False
            snapshot_metadatas.append(snapshot_metadata)

        previous_id_set = set(snapshot_ids)

        def restore_previous_version() -> None:
            """Restore the previous committed rows after an ambiguous write failure."""
            if snapshot_ids:
                self.collection.upsert(
                    ids=snapshot_ids,
                    documents=snapshot_documents,
                    embeddings=snapshot_embeddings,
                    metadatas=snapshot_metadatas,
                )
            # Remove newly introduced final IDs that did not exist in the previous
            # version (for example, chunks added when the document grew).
            new_final_ids = [chunk_id for chunk_id in final_ids if chunk_id not in previous_id_set]
            if new_final_ids:
                self.collection.delete(ids=new_final_ids)

        # Remove prior committed and staging rows for this URL only after the new
        # staged write has completed. The committed snapshot above is retained in
        # memory until promotion is confirmed.
        old_ids = [
            chunk_id
            for chunk_id, metadata in self._existing_chunks.items()
            if (metadata or {}).get("url") == document.url
        ]
        try:
            if old_ids:
                self.collection.delete(ids=old_ids)
        except Exception as delete_exc:
            try:
                restore_previous_version()
                self._existing_chunks = get_existing_chunks(self.collection)
            except Exception as restore_exc:
                raise RuntimeError(
                    "Failed to replace prior document chunks and failed to restore the "
                    "previous committed version. Temporary staged rows are marked and "
                    "excluded from search; rerun indexing to reconcile the document."
                ) from restore_exc
            raise RuntimeError(
                "Could not remove prior document chunks; the previous committed version "
                "was restored. Retry indexing after the Chroma delete failure is resolved."
            ) from delete_exc

        # Promotion writes deterministic IDs. A partial failure may have mutated
        # some final IDs, so retry once and restore the old snapshot if both attempts fail.
        try:
            upsert_chunks(
                self.collection,
                chunks,
                embeddings,
                embedding_model=self.embedding_model,
                ids=final_ids,
                embedding_dim=self.embedding_dim,
            )
        except Exception:
            try:
                upsert_chunks(
                    self.collection,
                    chunks,
                    embeddings,
                    embedding_model=self.embedding_model,
                    ids=final_ids,
                    embedding_dim=self.embedding_dim,
                )
            except Exception as promotion_exc:
                try:
                    restore_previous_version()
                except Exception as restore_exc:
                    try:
                        self._existing_chunks = get_existing_chunks(self.collection)
                    except Exception:
                        pass
                    raise RuntimeError(
                        "Chroma promotion failed twice and the previous committed "
                        "version could not be restored. Temporary stage rows remain "
                        "marked and excluded from search; investigate Chroma and rerun "
                        "indexing to reconcile the document."
                    ) from restore_exc
                self._existing_chunks = get_existing_chunks(self.collection)
                raise RuntimeError(
                    "Chroma upsert failed and the retry failed too. The previous "
                    "committed version was restored; temporary staging rows remain "
                    "hidden from ordinary search and can be reconciled by rerunning "
                    "indexing after the Chroma failure is resolved."
                ) from promotion_exc

        # If stage cleanup fails, committed final rows remain searchable and the next
        # run can discover/reconcile the marked stage rows.
        try:
            self.collection.delete(ids=staging_ids)
        except Exception:
            self._existing_chunks = get_existing_chunks(self.collection)
            raise
        self._drop_cached_chunks(document.url)
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
                "embedding_dim": self.embedding_dim,
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
            if (
                self.embedding_dim is not None
                and metadata.get("embedding_dim") != self.embedding_dim
            ):
                return False
        return True


def _chunk_id_for(chunk: Chunk) -> str:
    return _chunk_id(chunk)
