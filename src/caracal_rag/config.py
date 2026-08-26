import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    api_base: str = "http://localhost:3001/v1/embeddings"


@dataclass
class ChromaConfig:
    host: str = "localhost"
    port: int = 3400
    collection: str = "caracal-base"


@dataclass
class SourceConfig:
    name: str
    type: str  # "markdown", "github_md_doc_dir", "local_path"
    urls: List[str] = field(default_factory=list)
    repo: Optional[str] = None
    path: Optional[str] = None
    branch: Optional[str] = None


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    sources: List[SourceConfig] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: str) -> "Config":
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Load environment variables if available
        if "CHROMA_HOST" in os.environ:
            data["chroma"]["host"] = os.environ["CHROMA_HOST"]
        if "CHROMA_PORT" in os.environ:
            data["chroma"]["port"] = int(os.environ["CHROMA_PORT"])
        if "CHROMA_COLLECTION" in os.environ:
            data["chroma"]["collection"] = os.environ["CHROMA_COLLECTION"]
        if "EMBEDDING_API_BASE" in os.environ:
            data["embedding"]["api_base"] = os.environ["EMBEDDING_API_BASE"]

        embedding = EmbeddingConfig(**data.get("embedding", {}))
        chroma = ChromaConfig(**data.get("chroma", {}))

        sources = []
        for source_data in data.get("sources", []):
            source = SourceConfig(**source_data)
            sources.append(source)

        return cls(embedding=embedding, chroma=chroma, sources=sources)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding.__dict__,
            "chroma": self.chroma.__dict__,
            "sources": [source.__dict__ for source in self.sources]
        }

    def save(self, config_path: str):
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)