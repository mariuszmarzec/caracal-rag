from __future__ import annotations

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
        self.client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        self.collection = ensure_collection(self.client, config.chroma.collection)

    def _load_documents(self) -> Iterable[Document]:
        for doc in load_documents_from_config("config/sources.example.yaml"):
            if self.source_filter and doc.source != self.source_filter:
                continue
            yield doc

    def run(self) -> None:
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
        # content hash, there is nothing to embed or upsert.
        existing = get_existing_chunks(self.collection)
        if self._is_unchanged(document, existing):
            return

        delete_document_chunks(self.collection, document.url)
        chunks = list(chunk_document(document))
        if not chunks:
            return
        # A document whose content is whitespace-only still yields one empty
        # chunk via chunk_markdown; treat that as no content so stale chunks are
        # removed instead of an empty chunk being upserted.
        if all(not chunk.text.strip() for chunk in chunks):
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(
            texts,
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
        )
        upsert_chunks(self.collection, chunks, embeddings)

    @staticmethod
    def _is_unchanged(document: Document, existing: dict[str, dict]) -> bool:
        """Return True when every chunk for this document already exists and matches.

        Compares the expected chunk-id set against the ids actually stored for
        this document, so a shrinking chunk count (or a document that became
        empty) is detected as a change and the document is refreshed. Checking
        only the new chunks would leave stale vectors behind when the chunk
        layout changes while the text stays the same.
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
        return True


def _chunk_id_for(chunk: Chunk) -> str:
    # Local import to avoid a circular dependency at module import time.
    from caracal_rag.vectorstore import _chunk_id

    return _chunk_id(chunk)
