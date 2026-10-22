from __future__ import annotations

import numpy as np

from caracal_rag.vectorstore import collection_embedding_dim


class _NumpyEmbeddingCollection:
    def __init__(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        self.embeddings = embeddings
        self.metadatas = metadatas

    def get(self, include=None):
        return {"metadatas": self.metadatas, "embeddings": self.embeddings}


def test_collection_embedding_dim_handles_numpy_embeddings() -> None:
    """Chroma returns NumPy arrays for included embeddings in recent versions."""
    collection = _NumpyEmbeddingCollection(
        embeddings=np.zeros((1, 1024), dtype=np.float32),
        metadatas=[{"embedding_model": "bge-m3"}],
    )

    assert collection_embedding_dim(collection) == 1024


def test_collection_embedding_dim_handles_empty_numpy_embeddings() -> None:
    collection = _NumpyEmbeddingCollection(
        embeddings=np.empty((0, 1024), dtype=np.float32),
        metadatas=[],
    )

    assert collection_embedding_dim(collection) is None
