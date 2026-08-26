from __future__ import annotations

from typing import Iterable

from caracal_rag.chunking import chunk_document
from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.sources import Document, load_documents_from_config
from caracal_rag.vectorstore import (
    _chroma_client,
    delete_document_chunks,
    ensure_collection,
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
        documents = list(self._load_documents())
        for document in documents:
            self._index_document(document)

    def _index_document(self, document: Document) -> None:
        delete_document_chunks(self.collection, document.url)
        chunks = list(chunk_document(document))
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(
            texts,
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
        )
        upsert_chunks(self.collection, chunks, embeddings)
