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



def test_e5_model_uses_query_and_passage_prefixes(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(embeddings.litellm, "api_base", None, raising=False)
    monkeypatch.setattr(embeddings.litellm, "api_key", None, raising=False)

    embeddings.embed_texts(
        ["document text"],
        api_base="http://127.0.0.1:4000/v1",
        model="multilingual-e5-base",
        api_key="test-key",
        input_type="passage",
    )
    embeddings.embed_texts(
        ["search terms"],
        api_base="http://127.0.0.1:4000/v1",
        model="qllama/multilingual-e5-base",
        api_key="test-key",
        input_type="query",
    )

    assert calls[0]["input"] == ["passage: document text"]
    assert calls[1]["input"] == ["query: search terms"]


def test_e5_prefix_replaces_an_existing_opposite_prefix() -> None:
    assert embeddings._prepare_embedding_text(
        "passage: document", "multilingual-e5-base", "query"
    ) == "query: document"
    assert embeddings._prepare_embedding_text(
        "query: question", "multilingual-e5-base", "passage"
    ) == "passage: question"


def test_non_e5_models_are_not_prefixed() -> None:
    assert embeddings._prepare_embedding_text("hello", "bge-m3", "query") == "hello"
