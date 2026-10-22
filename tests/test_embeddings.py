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
