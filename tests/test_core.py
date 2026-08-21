from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

mcp_stub = types.ModuleType("mcp")
mcp_server_stub = types.ModuleType("mcp.server")
mcp_types_stub = types.ModuleType("mcp.types")
mcp_server_stub.Server = object
mcp_types_stub.Tool = object
mcp_types_stub.TextContent = object
mcp_stub.server = mcp_server_stub
mcp_stub.types = mcp_types_stub
sys.modules["mcp"] = mcp_stub
sys.modules["mcp.server"] = mcp_server_stub
sys.modules["mcp.types"] = mcp_types_stub

litellm_stub = types.ModuleType("litellm")
class FakeLiteLLM:
    api_base = None
    api_key = None
    @staticmethod
    def embedding(model: str, input: list[str]) -> dict:
        return {"data": [{"embedding": [0.1, 0.2]}]}
litellm_stub.embedding = FakeLiteLLM.embedding
litellm_stub.api_base = None
litellm_stub.api_key = None
sys.modules["litellm"] = litellm_stub

from caracal_rag.chunking import chunk_document, chunk_markdown, infer_type  # noqa: E402
from caracal_rag.config import AppConfig, SourceConfig  # noqa: E402
from caracal_rag.indexing import Indexer  # noqa: E402
from caracal_rag.mcp import CaracalMcpServer, SearchResult  # noqa: E402
from caracal_rag.sources import Document  # noqa: E402


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    data = {
        "embedding": {"model": "mock", "api_base": "http://localhost:4000"},
        "chroma": {"host": "localhost", "port": 8000, "collection": "test"},
        "sources": [
            {
                "name": "api-docs",
                "type": "markdown",
                "urls": ["https://example.com/login.md"],
            }
        ],
    }
    config_path = tmp_path / "sources.example.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_config_parsing(example_config: Path) -> None:
    config = AppConfig.from_yaml(str(example_config))
    assert len(config.sources) == 1
    assert config.sources[0].name == "api-docs"
    assert config.chroma.collection == "test"
    assert config.embedding.model == "mock"


def test_infer_type() -> None:
    assert infer_type("login.md") == "markdown"
    assert infer_type("data.json") == "json"
    assert infer_type("setup.yml") == "yaml"


def test_chunk_markdown() -> None:
    text = "# Title\n\nBody paragraph.\n\n## Section\n\nMore content.\n"
    chunks = chunk_markdown(text, max_chars=10)
    assert len(chunks) >= 1
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_chunk_document() -> None:
    doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login\n\nDetails here.",
        content_hash=Document.hash("# Login\n\nDetails here."),
    )
    chunks = list(chunk_document(doc))
    assert len(chunks) >= 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content_hash == Document.hash(doc.content)


def test_content_hash() -> None:
    assert Document.hash("a") == Document.hash("a")
    assert Document.hash("a") != Document.hash("b")


def test_indexing_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeChroma:
        def __init__(self) -> None:
            self.calls = []

        def get_or_create_collection(self, name: str) -> "FakeCollection":
            return FakeCollection()

    class FakeCollection:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.added = 0

        def add(self, **kwargs) -> None:
            self.added += 1

        def get(self, where: dict) -> dict:
            return {"ids": ["url1-chunk0"]}

        def delete(self, ids: list[str]) -> None:
            self.deleted.extend(ids)

    monkeypatch.setenv("CARACAL_EMBEDDING_MODEL", "mock")
    monkeypatch.setenv("CARACAL_EMBEDDING_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("CARACAL_CHROMA_HOST", "localhost")
    monkeypatch.setenv("CARACAL_CHROMA_PORT", "8000")
    monkeypatch.setenv("CARACAL_CHROMA_COLLECTION", "test")

    monkeypatch.setattr("caracal_rag.indexing._chroma_client", lambda *args, **kwargs: FakeChroma())
    monkeypatch.setattr("caracal_rag.indexing.embed_texts", lambda texts, **kwargs: [[0.1, 0.2] for _ in texts])

    fake_doc = Document(
        source="api",
        name="login.md",
        url="https://example.com/login.md",
        type="markdown",
        content="# Login",
        content_hash=Document.hash("# Login"),
    )
    monkeypatch.setattr("caracal_rag.indexing.load_documents_from_config", lambda path: [fake_doc])

    config = AppConfig(
        sources=[SourceConfig(name="api", type="markdown", urls=["https://example.com/login.md"])],
        embedding=AppConfig.from_yaml("config/sources.example.yaml").embedding,
        chroma=AppConfig.from_yaml("config/sources.example.yaml").chroma,
    )
    indexer = Indexer(config=config)
    indexer.run()
    assert isinstance(indexer.collection, FakeCollection)
