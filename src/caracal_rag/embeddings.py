"""Embedding generation through a LiteLLM-compatible (FreeLLMAPI) endpoint.

The embedding model is *never* defaulted here: callers must pass an explicit
model taken from the active configuration. Silently choosing a model would let
an index be built in an embedding space the user did not pick, and would make
stored vectors incompatible with later queries.
"""

from __future__ import annotations

import os

import litellm

#: Suffix LiteLLM/the OpenAI SDK appends when calling ``embeddings.create()``.
EMBEDDINGS_SUFFIX = "/embeddings"


def _normalize_api_base(api_base: str) -> str:
    """Return a base URL LiteLLM can use without double-appending ``/embeddings``.

    LiteLLM passes ``api_base`` to the OpenAI SDK as ``base_url``. The OpenAI SDK
    appends ``/embeddings`` when calling ``embeddings.create()``, so a configured
    value of ``http://localhost:3001/v1/embeddings`` produces a malformed request
    to ``/v1/embeddings/embeddings``. Strip a trailing ``/embeddings`` so both
    ``http://localhost:3001/v1`` and ``http://localhost:3001/v1/embeddings`` work.
    """
    base = api_base.rstrip("/")
    if base.endswith(EMBEDDINGS_SUFFIX):
        return base[: -len(EMBEDDINGS_SUFFIX)]
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
    model: str,
    api_key: str | None = None,
) -> list[list[float]]:
    """Embed ``texts`` with the explicitly configured ``model``.

    ``model`` is required: there is no built-in fallback, because a silently
    chosen family would make the index incompatible with the vectors the user
    expects.
    """
    if not str(model or "").strip():
        raise ValueError(
            "embed_texts requires an explicit embedding model; none was configured."
        )
    if str(model).strip().lower() == "auto":
        raise ValueError(
            "The embedding model 'auto' is unsafe for a persistent index because the "
            "effective FreeLLMAPI family may change. Configure an explicit family/model."
        )
    _get_embedding_client(api_base=api_base, api_key=api_key)
    vectors: list[list[float]] = []
    for text in texts:
        try:
            response = litellm.embedding(input=[text], model=model)
        except Exception as exc:  # noqa: BLE001 - sanitize service diagnostics
            # Provider clients sometimes include request details in exceptions.
            # Never allow a configured key to leak from index/check/MCP errors.
            safe_message = redact_secret(str(exc), api_key)
            raise RuntimeError(f"Embedding request failed: {safe_message}") from None
        vectors.append(response.data[0]["embedding"])
    return vectors


def redact_secret(text: str, secret: str | None) -> str:
    """Return ``text`` with any occurrence of ``secret`` replaced by ``***``.

    Error messages must never echo the FreeLLMAPI key, so diagnostics redact it
    before being printed or logged.
    """
    if not secret:
        return text
    return text.replace(secret, "***")


def validate_embedding_endpoint(
    api_base: str,
    model: str,
    api_key: str | None = None,
) -> None:
    """Probe the configured embedding endpoint and raise ``ConfigError`` on failure.

    A single embedding call is issued with the configured model. Failures are
    wrapped in an actionable message that names the endpoint and model without
    ever exposing the API key.
    """
    from caracal_rag.config import ConfigError

    if not str(model or "").strip():
        raise ConfigError(
            "No embedding model is configured. Set `embedding.model` in the "
            "active configuration (for example `embedding.model: bge-m3`) after "
            "checking what is enabled under FreeLLMAPI Models -> Embeddings."
        )
    try:
        embed_texts(["caracal-rag check"], api_base=api_base, model=model, api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - surface any connectivity/model failure
        message = redact_secret(str(exc), api_key)
        raise ConfigError(
            f"Embedding endpoint at {_normalize_api_base(api_base)!r} could not "
            f"serve model {model!r}: {message}. Verify the FreeLLMAPI service is "
            "running, the base URL is correct, and that model is enabled under "
            "FreeLLMAPI Models -> Embeddings."
        ) from exc

