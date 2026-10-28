from __future__ import annotations

from types import SimpleNamespace

from caracal_rag import embeddings


def test_embed_texts_uses_openai_provider_for_freellmapi(
    monkeypatch,
) -> None:
    """Route OpenAI-compatible FreeLLMAPI calls through LiteLLM's OpenAI adapter."""
    calls: list[dict] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(embeddings.litellm, "api_base", None, raising=False)
    monkeypatch.setattr(embeddings.litellm, "api_key", None, raising=False)

    vectors = embeddings.embed_texts(
        ["hello"],
        api_base="http://127.0.0.1:3001/v1",
        model="bge-m3",
        api_key="test-key",
    )

    assert vectors == [[0.1, 0.2, 0.3]]
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/bge-m3"
    assert calls[0]["input"] == ["hello"]
    assert embeddings.litellm.api_base == "http://127.0.0.1:3001/v1"
    assert embeddings.litellm.api_key == "test-key"


def test_multilingual_e5_adds_query_prefix(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(embeddings.litellm, "api_base", None, raising=False)
    monkeypatch.setattr(embeddings.litellm, "api_key", None, raising=False)

    embeddings.embed_texts(
        ["jak działa paginacja"],
        api_base="http://127.0.0.1:4000/v1",
        model="multilingual-e5-base",
        api_key="test-key",
        input_type="query",
    )

    assert calls[0]["model"] == "openai/multilingual-e5-base"
    assert calls[0]["input"] == ["query: jak działa paginacja"]


def test_multilingual_e5_adds_passage_prefix_for_qualified_model(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(embeddings.litellm, "api_base", None, raising=False)
    monkeypatch.setattr(embeddings.litellm, "api_key", None, raising=False)

    embeddings.embed_texts(
        ["Dokument opisuje konfigurację."],
        api_base="http://127.0.0.1:4000/v1",
        model="ollama/qllama/multilingual-e5-base",
        api_key="test-key",
        input_type="passage",
    )

    assert calls[0]["input"] == ["passage: Dokument opisuje konfigurację."]


def test_other_embedding_models_keep_original_text(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(embeddings.litellm, "api_base", None, raising=False)
    monkeypatch.setattr(embeddings.litellm, "api_key", None, raising=False)

    embeddings.embed_texts(
        ["plain input"],
        api_base="http://127.0.0.1:4000/v1",
        model="bge-m3",
        api_key="test-key",
        input_type="passage",
    )

    assert calls[0]["input"] == ["plain input"]
