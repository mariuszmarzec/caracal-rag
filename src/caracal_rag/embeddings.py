from __future__ import annotations

import os

import litellm


def _get_embedding_client(api_base: str, api_key: str | None = None) -> None:
    litellm.api_base = api_base
    if api_key:
        litellm.api_key = api_key
    else:
        litellm.api_key = os.getenv("LITELLM_API_KEY", None)


def embed_texts(
    texts: list[str], api_base: str, api_key: str | None = None
) -> list[list[float]]:
    _get_embedding_client(api_base=api_base, api_key=api_key)
    vectors: list[list[float]] = []
    for text in texts:
        response = litellm.embedding(input=[text])
        vectors.append(response.data[0]["embedding"])
    return vectors
