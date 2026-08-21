from __future__ import annotations

import os
from typing import Iterable


def get_embedding_client(api_base: str, api_key: str | None = None) -> None:
    import litellm

    litellm.api_base = api_base
    if api_key:
        litellm.api_key = api_key


def embed_texts(texts: list[str], model: str, api_base: str, api_key: str | None = None) -> list[list[float]]:
    import litellm

    get_embedding_client(api_base=api_base, api_key=api_key)
    vectors: list[list[float]] = []
    for text in texts:
        response = litellm.embedding(model=model, input=[text])
        vectors.append(response.data[0]["embedding"])
    return vectors
