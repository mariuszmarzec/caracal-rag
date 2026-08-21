from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class SourceConfig:
    name: str
    type: str
    urls: list[str] | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if not self.urls and not self.url:
            raise ValueError(f"Source {self.name!r} must define urls or url")


@dataclass
class EmbeddingConfig:
    model: str = os.getenv("CARACAL_EMBEDDING_MODEL", "text-embedding-3-small")
    api_base: str = os.getenv("CARACAL_EMBEDDING_API_BASE", "http://localhost:4000")
    api_key: str | None = os.getenv("CARACAL_EMBEDDING_API_KEY", None)


@dataclass
class ChromaConfig:
    host: str = os.getenv("CARACAL_CHROMA_HOST", "localhost")
    port: int = int(os.getenv("CARACAL_CHROMA_PORT", "8000"))
    collection: str = os.getenv("CARACAL_CHROMA_COLLECTION", "caracal")
    ssl: bool = os.getenv("CARACAL_CHROMA_SSL", "false").lower() in {"1", "true", "yes"}


@dataclass
class AppConfig:
    sources: list[SourceConfig]
    embedding: EmbeddingConfig
    chroma: ChromaConfig

    @classmethod
    def from_yaml(cls, path: str) -> AppConfig:
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        sources = [SourceConfig(**item) for item in data.get("sources", [])]
        embedding_data = data.get("embedding", {}) or {}
        chroma_data = data.get("chroma", {}) or {}

        embedding = EmbeddingConfig(**embedding_data)
        chroma = ChromaConfig(**chroma_data)

        return cls(sources=sources, embedding=embedding, chroma=chroma)
