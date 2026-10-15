from __future__ import annotations

import os

import litellm


def _normalize_api_base(api_base: str) -> str:
    """Return a base URL LiteLLM can use without double-appending ``/embeddings``.

    LiteLLM passes ``api_base`` to the OpenAI SDK as ``base_url``. The OpenAI SDK
    appends ``/embeddings`` when calling ``embeddings.create()``, so a configured
    value of ``http://localhost:3001/v1/embeddings`` produces a malformed request
    to ``/v1/embeddings/embeddings``. Strip a trailing ``/embeddings`` so both
    ``http://localhost:3001/v1`` and ``http://localhost:3001/v1/embeddings`` work.
    """
    base = api_base.rstrip("/")
    if base.endswith("/embeddings"):
        return base[: -len("/embeddings")]
    return base


def _get_embedding_client(api_base: str, api_key: str | None = None) -> None:
    litellm.api_base = _normalize_api_base(api_base)
    if api_key:
        litellm.api_key = api_key
    else:
        litellm.api_key = os.getenv("LITELLM_API_KEY", None)


def embed_texts(
    texts: list[str],
    api_base: str,
    api_key: str | None = None,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    _get_embedding_client(api_base=api_base, api_key=api_key)
    vectors: list[list[float]] = []
    for text in texts:
        response = litellm.embedding(input=[text], model=model)
        vectors.append(response.data[0]["embedding"])
    return vectors
