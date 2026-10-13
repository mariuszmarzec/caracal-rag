"""Tests for the MCP tools: search_api and find_endpoint.

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
    return CaracalMcpServer(AppConfig.from_yaml("config/sources.example.yaml"))


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


def test_find_endpoint_match(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    results = server.find_endpoint(path="/fiteo/api/1/login")
    assert len(results) == 1
    assert results[0].document == "login.md"


def test_find_endpoint_normalises_method_and_slashes(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    assert server.find_endpoint(path="POST /fiteo/api/1/login/") == server.find_endpoint(
        path="/fiteo/api/1/login"
    )


def test_find_endpoint_no_match_reports_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    results = server.find_endpoint(path="/fiteo/api/1/does-not-exist")
    assert results == []
    text = server._format_endpoint(results, "/fiteo/api/1/does-not-exist", None)
    assert "No match" in text


def test_find_endpoint_empty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    assert server.find_endpoint(path="") == []


def test_find_endpoint_respects_method_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    # The fake collection returns a login.md chunk; filtering by GET must not match.
    results = server.find_endpoint(path="/fiteo/api/1/login", method="GET")
    assert results == []


def test_list_tools_exposes_both_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    server = _make_server(monkeypatch)
    # The list_tools handler is registered as a coroutine; call it directly to assert
    # the tool definitions.
    handler = server.server.list_tools_handler
    assert handler is not None
    tools = asyncio.run(handler())
    names = [t.name for t in tools]
    assert "search_api" in names
    assert "find_endpoint" in names
