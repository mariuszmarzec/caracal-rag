from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from chromadb import HttpClient

from caracal_rag.chunking import Chunk


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict


def _chroma_client(host: str, port: int, ssl: bool) -> HttpClient:
    return HttpClient(host=host, port=int(port), ssl=bool(ssl))


def ensure_collection(client: HttpClient, name: str) -> any:
    return client.get_or_create_collection(name=name)


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic chunk id derived from the document url and chunk index.

    Using a stable id makes repeated indexing idempotent: re-upserting the same
    chunk overwrites the previous vector instead of duplicating it.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.url}-{chunk.chunk_index}"))


def upsert_chunks(collection, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
    """Insert or update chunk vectors.

    Chunks are keyed by a deterministic id (document url + chunk index), so
    re-running the index on unchanged documents overwrites existing vectors
    instead of appending duplicates.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for idx, chunk in enumerate(chunks):
        ids.append(_chunk_id(chunk))
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "document": chunk.document,
                "url": chunk.url,
                "type": chunk.type,
                "content_hash": chunk.content_hash,
                "chunk_index": chunk.chunk_index,
            }
        )
    if not ids:
        return
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_document_chunks(collection, document_url: str) -> None:
    """Delete all chunks belonging to a document url."""
    results = collection.get(where={"url": document_url}, include=["metadatas"])
    ids = results.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def delete_missing_documents(
    collection,
    current_urls: Iterable[str],
    source: str | None = None,
) -> list[str]:
    """Remove chunks whose document url is no longer present in the source.

    Returns the list of removed document urls. This keeps the index consistent
    when source documents are deleted between indexing runs.

    When ``source`` is provided, only chunks from that source are considered
    for deletion, so indexing a single source never removes vectors from other
    configured sources.
    """
    current = set(current_urls)
    # Request metadatas explicitly: Chroma does not return metadata unless it is
    # asked for, so stale vectors would otherwise go undetected.
    existing = collection.get(include=["metadatas"])
    ids = existing.get("ids") or []
    metadatas = existing.get("metadatas") or []

    stale_urls: set[str] = set()
    stale_ids: list[str] = []
    for chunk_id, metadata in zip(ids, metadatas):
        url = (metadata or {}).get("url")
        if url and url not in current:
            if source is not None and (metadata or {}).get("source") != source:
                continue
            stale_urls.add(url)
            stale_ids.append(chunk_id)

    if stale_ids:
        collection.delete(ids=stale_ids)
    return sorted(stale_urls)


def get_existing_chunks(collection) -> dict[str, dict]:
    """Return a mapping of chunk id -> metadata for every chunk in the collection."""
    existing = collection.get(include=["metadatas"])
    ids = existing.get("ids") or []
    metadatas = existing.get("metadatas") or []
    return {chunk_id: metadata or {} for chunk_id, metadata in zip(ids, metadatas)}


def search_chunks(collection, query_embedding: list[float], top_k: int = 5):
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
