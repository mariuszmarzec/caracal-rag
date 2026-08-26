from __future__ import annotations

import sys
import types

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
    def embedding(*args, **kwargs) -> dict:
        return {"data": [{"embedding": [0.1, 0.2]}]}
litellm_stub.embedding = FakeLiteLLM.embedding
litellm_stub.api_base = None
litellm_stub.api_key = None
sys.modules["litellm"] = litellm_stub

from caracal_rag.config import AppConfig  # noqa: E402
from caracal_rag.mcp import CaracalMcpServer, SearchResult  # noqa: E402


class FakeChroma:
    def __init__(self) -> None:
        pass

    def get_or_create_collection(self, name: str) -> "FakeCollection":
        return FakeCollection()


class FakeCollection:
    def __init__(self) -> None:
        pass

    def query(self, query_embeddings, n_results: int = 5) -> dict:
        return {
            "documents": [["Login details."]],
            "metadatas": [[{"source": "api", "document": "login.md", "url": "https://example.com/login.md"}]],
        }


class FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name

    def list_tools(self):
        def decorator(func):
            return func
        return decorator

    def call_tool(self):
        def decorator(func):
            return func
        return decorator


def test_search_formatting(monkeypatch) -> None:
    monkeypatch.setattr("caracal_rag.mcp._chroma_client", lambda *args, **kwargs: FakeChroma())
    monkeypatch.setattr("caracal_rag.mcp.embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr("caracal_rag.mcp.Server", FakeServer)
    results = [
        SearchResult(
            source="api",
            document="login.md",
            url="https://example.com/login.md",
            content="Login details.",
        )
    ]
    server = CaracalMcpServer(AppConfig.from_yaml("config/sources.example.yaml"))
    text = server._format(results)
    assert "Source: api" in text
    assert "Document: login.md" in text
    assert "Login details." in text
