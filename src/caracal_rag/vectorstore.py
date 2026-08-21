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


def upsert_chunks(collection, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for idx, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.url}-{chunk.chunk_index}"))
        ids.append(chunk_id)
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
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_document_chunks(collection, document_url: str) -> None:
    results = collection.get(where={"url": document_url})
    ids = results.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def search_chunks(collection, query_embedding: list[float], top_k: int = 5):
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
