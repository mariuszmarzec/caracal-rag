import os
from typing import List, Optional
from caracal_rag.config import EmbeddingConfig


class EmbeddingsGenerator:
    """LiteLLM-backed embeddings abstraction."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.config.api_base,
                api_key=os.environ.get("LITELLM_API_KEY", "sk-litellm")
            )
        return self._client

    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        response = self.client.embeddings.create(
            model=self.config.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def generate_chunks(self, chunks: List[dict]) -> List[dict]:
        """Generate embeddings for a list of chunk dicts."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    def get_model_name(self) -> str:
        return self.config.model