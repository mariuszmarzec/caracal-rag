"""Tests for the MCP tools: search_api.

External services (LiteLLM, Chroma) are replaced with in-memory fakes so the
tests run without network access.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# Stub out heavy optional dependencies so importing caracal_rag.mcp works in CI.
chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

mcp_stub = types.ModuleType("mcp")
mcp_server_stub = types.ModuleType("mcp.server")
mcp_types_stub = types.ModuleType("mcp.types")
mcp_server_stub.Server = object
class _ToolStub:
    def __init__(self, *args, **kwargs) -> None:
        self.name = kwargs.get("name", "")
        self.description = kwargs.get("description", "")
        self.inputSchema = kwargs.get("inputSchema", {})


class _TextContentStub:
    def __init__(self, *args, **kwargs) -> None:
        self.type = kwargs.get("type", "text")
        self.text = kwargs.get("text", "")


mcp_types_stub.Tool = _ToolStub
mcp_types_stub.TextContent = _TextContentStub
mcp_stub.server = mcp_server_stub
mcp_stub.types = mcp_types_stub
sys.modules["mcp"] = mcp_stub
sys.modules["mcp.server"] = mcp_server_stub
sys.modules["mcp.types"] = mcp_types_stub

litellm_stub = types.ModuleType("litellm")


class _FakeLiteLLM:
    api_base = None
    api_key = None

    @staticmethod
    def embedding(*args, **kwargs) -> dict:
        return {"data": [{"embedding": [0.1, 0.2]}]}


litellm_stub.embedding = _FakeLiteLLM.embedding
litellm_stub.api_base = None
litellm_stub.api_key = None
sys.modules["litellm"] = litellm_stub

from caracal_rag.config import AppConfig  # noqa: E402
from caracal_rag.mcp import CaracalMcpServer, SearchResult  # noqa: E402


class FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.list_tools_handler = None
        self.call_tool_handler = None

    def list_tools(self):
        def decorator(func):
            self.list_tools_handler = func
            return func
        return decorator

    def call_tool(self):
        def decorator(func):
            self.call_tool_handler = func
            return func
        return decorator


class FakeChroma:
    def __init__(self) -> None:
        pass

    def get_or_create_collection(self, name: str) -> "FakeCollection":
        return FakeCollection()


class FakeCollection:
    def __init__(self) -> None:
        pass

    def get(
        self,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        # Empty collection: no stored vectors, so the embedding-model consistency
        # check passes and the query below is unaffected.
        return {"ids": [], "metadatas": []}

    def query(self, query_embeddings, n_results: int = 5) -> dict[str, Any]:
        return {
            "documents": [["# POST /fiteo/api/1/login\n\nLogs in with email and password."]],
            "metadatas": [
                [
                    {
                        "source": "fiteo-docs",
                        "document": "login.md",
                        "url": "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/login.md",
                    }
                ]
            ],
        }


def _make_server(monkeypatch: pytest.MonkeyPatch) -> CaracalMcpServer:
    monkeypatch.setattr("caracal_rag.mcp._chroma_client", lambda *args, **kwargs: FakeChroma())
    monkeypatch.setattr("caracal_rag.mcp.embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr("caracal_rag.mcp.Server", FakeServer)
    config = AppConfig.from_yaml("config/sources.example.yaml")
    # The protected interface requires a valid token; tests supply one directly
    # on the config object rather than through the environment/secrets layer.
    config.access_token.token = "test-token"
    return CaracalMcpServer(config)


def test_search_api_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    results = server.search(query="login", top_k=5)
    assert len(results) == 1
    item = results[0]
    assert isinstance(item, SearchResult)
    assert item.source == "fiteo-docs"
    assert item.document == "login.md"
    assert item.url == (
        "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/login.md"
    )
    assert item.content.startswith("# POST /fiteo/api/1/login")


def test_search_api_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    results = [
        SearchResult(
            source="fiteo-docs",
            document="login.md",
            url="https://example.com/login.md",
            content="Login details.",
        )
    ]
    text = server._format(results)
    assert "Source: fiteo-docs" in text
    assert "Document: login.md" in text
    assert "Login details." in text


def test_list_tools_exposes_only_search_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    server = _make_server(monkeypatch)
    # The list_tools handler is registered as a coroutine; call it directly to assert
    # the tool definitions.
    handler = server.server.list_tools_handler
    assert handler is not None
    tools = asyncio.run(handler())
    names = [t.name for t in tools]
    assert names == ["search_api"]


def test_call_tool_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(ValueError, match="Unknown tool"):
        asyncio.run(handler("bogus", {"access_token": "test-token"}))


def test_parse_results_returns_all_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """_parse_results must flatten the nested Chroma response and return every
    top-k chunk, not merely the first nested item.

    Chroma nests results as documents=[[...top-k texts...]] and
    metadatas=[[...top-k metadata...]]. A parser that iterates the outer arrays
    and selects element [0] discards all but one result for top_k > 1.
    """
    response = {
        "documents": [
            ["# POST /fiteo/api/1/login\n\nLogs in.", "# GET /fiteo/api/1/tasks\n\nLists tasks."]
        ],
        "metadatas": [
            [
                {
                    "source": "fiteo-docs",
                    "document": "login.md",
                    "url": "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/login.md",
                },
                {
                    "source": "fiteo-docs",
                    "document": "tasks.md",
                    "url": "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/tasks.md",
                },
            ]
        ],
    }
    results = CaracalMcpServer._parse_results(response)
    assert len(results) == 2
    assert results[0].document == "login.md"
    assert results[0].url.endswith("docs/api/login.md")
    assert results[1].document == "tasks.md"
    assert results[1].url.endswith("docs/api/tasks.md")
    assert results[0].content.startswith("# POST /fiteo/api/1/login")
    assert results[1].content.startswith("# GET /fiteo/api/1/tasks")


def test_search_raises_on_embedding_model_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3: search must refuse to query when stored vectors use a different model.

    The consistency guard runs before any embedding/querying so a mismatched
    index is surfaced with actionable rebuild guidance rather than producing
    meaningless results.
    """
    class MismatchCollection:
        def get(self, where=None, include=None):
            return {
                "ids": ["chunk-1"],
                "metadatas": [{"embedding_model": "text-embedding-3-large"}],
            }

        def query(self, query_embeddings, n_results: int = 5):
            return {"documents": [], "metadatas": [], "ids": [], "distances": []}

    monkeypatch.setattr("caracal_rag.mcp.Server", FakeServer)

    class _ChromaWithMismatch:
        def get_or_create_collection(self, name: str) -> MismatchCollection:
            return MismatchCollection()

    monkeypatch.setattr(
        "caracal_rag.mcp._chroma_client",
        lambda *args, **kwargs: _ChromaWithMismatch(),
    )
    monkeypatch.setattr("caracal_rag.mcp.embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])

    server = CaracalMcpServer(AppConfig.from_yaml("config/sources.example.yaml"))
    with pytest.raises(ValueError, match="Mixing embedding spaces"):
        server.search(query="login", top_k=5)


def test_search_refuses_mixed_collection_with_untagged_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search must refuse a collection holding one tagged and one untagged vector.

    A mixed collection has vectors from an unknown embedding space alongside
    tagged ones; querying it would return untagged chunks mixed with query
    vectors from the configured model. The consistency guard must raise with
    rebuild guidance before any embedding or query is issued.
    """
    class MixedCollection:
        def get(self, where=None, include=None):
            return {
                "ids": ["chunk-tagged", "chunk-untagged"],
                "metadatas": [
                    {"embedding_model": "text-embedding-3-small"},
                    {"url": "https://example.com/unknown.md"},
                ],
            }

        def query(self, query_embeddings, n_results: int = 5):
            return {"documents": [], "metadatas": [], "ids": [], "distances": []}

    monkeypatch.setattr("caracal_rag.mcp.Server", FakeServer)

    class _ChromaWithMixed:
        def get_or_create_collection(self, name: str) -> MixedCollection:
            return MixedCollection()

    monkeypatch.setattr(
        "caracal_rag.mcp._chroma_client",
        lambda *args, **kwargs: _ChromaWithMixed(),
    )
    monkeypatch.setattr("caracal_rag.mcp.embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])

    server = CaracalMcpServer(AppConfig.from_yaml("config/sources.example.yaml"))
    with pytest.raises(ValueError, match="no embedding_model metadata"):
        server.search(query="login", top_k=5)
