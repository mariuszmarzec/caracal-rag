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
    repo: str | None = None
    path: str | None = None
    branch: str | None = None

    def __post_init__(self) -> None:
        has_urls = self.urls or self.url
        if self.type == "github_md_doc_dir":
            if not self.repo or not self.path:
                raise ValueError(
                    f"Source {self.name!r} of type github_md_doc_dir must define repo and path"
                )
        elif not has_urls:
            raise ValueError(
                f"Source {self.name!r} must define urls, url, or repo+path"
            )


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


def _coerce_source(item: dict[str, Any]) -> SourceConfig:
    """Coerce a raw source dict into a SourceConfig."""
    return SourceConfig(**item)


@dataclass
class AppConfig:
    sources: list[SourceConfig]
    embedding: EmbeddingConfig
    chroma: ChromaConfig

    @classmethod
    def from_yaml(cls, path: str) -> AppConfig:
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        sources = [_coerce_source(item) for item in data.get("sources", [])]
        embedding_data = data.get("embedding", {}) or {}
        chroma_data = data.get("chroma", {}) or {}

        embedding = EmbeddingConfig(**embedding_data)
        chroma = ChromaConfig(**chroma_data)

        return cls(sources=sources, embedding=embedding, chroma=chroma)
