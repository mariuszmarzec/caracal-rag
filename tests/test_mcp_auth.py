"""Tests for access-token enforcement on the protected MCP interface.

The MCP server is the intended protected Caracal RAG request path. Every tool
call must authenticate by default; missing or invalid tokens are denied and
authenticated calls retain the existing indexing/MCP behavior.
"""

from __future__ import annotations

import asyncio
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

from caracal_rag.auth import AuthenticationError  # noqa: E402
from caracal_rag.config import AppConfig  # noqa: E402
from caracal_rag.mcp import ACCESS_TOKEN_FIELD, CaracalMcpServer  # noqa: E402

VALID_TOKEN = "correct-horse-battery-staple"


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
    def get_or_create_collection(self, name: str) -> "FakeCollection":
        return FakeCollection()


class FakeCollection:
    def get(self, where=None, include=None):
        return {"ids": [], "metadatas": []}

    def query(self, query_embeddings, n_results: int = 5) -> dict[str, Any]:
        return {
            "documents": [["Login details."]],
            "metadatas": [[{"source": "api", "document": "login.md", "url": "https://example.com/login.md"}]],
        }


def _make_server(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    token: str | None = VALID_TOKEN,
) -> CaracalMcpServer:
    monkeypatch.setattr("caracal_rag.mcp._chroma_client", lambda *args, **kwargs: FakeChroma())
    monkeypatch.setattr("caracal_rag.mcp.embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr("caracal_rag.mcp.Server", FakeServer)
    config = AppConfig.from_yaml("config/sources.example.yaml")
    config.access_token.enabled = enabled
    config.access_token.token = token
    return CaracalMcpServer(config)


def test_tool_schema_requires_access_token_field(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    handler = server.server.list_tools_handler
    assert handler is not None
    tools = asyncio.run(handler())
    assert len(tools) == 1
    schema = tools[0].inputSchema
    assert ACCESS_TOKEN_FIELD in schema["properties"]
    assert ACCESS_TOKEN_FIELD in schema["required"]
    assert schema["properties"][ACCESS_TOKEN_FIELD]["type"] == "string"


def test_call_tool_denies_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(AuthenticationError, match="no access token was presented"):
        asyncio.run(handler("search_api", {"query": "login"}))


def test_call_tool_denies_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(AuthenticationError, match="does not match"):
        asyncio.run(
            handler("search_api", {"query": "login", ACCESS_TOKEN_FIELD: "wrong-token"})
        )


def test_call_tool_denies_short_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(AuthenticationError, match="too short"):
        asyncio.run(
            handler("search_api", {"query": "login", ACCESS_TOKEN_FIELD: "short"})
        )


def test_call_tool_accepts_valid_token_and_preserves_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authenticated call retains the existing indexing/MCP behavior."""
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    result = asyncio.run(
        handler("search_api", {"query": "login", ACCESS_TOKEN_FIELD: VALID_TOKEN})
    )
    assert len(result) == 1
    assert result[0].type == "text"
    assert "Source: api" in result[0].text
    assert "Document: login.md" in result[0].text


def test_call_tool_unknown_tool_still_denied_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth runs before tool dispatch, so unknown tools are also rejected."""
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(AuthenticationError, match="no access token was presented"):
        asyncio.run(handler("bogus", {}))


def test_disabled_policy_skips_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch, enabled=False, token=None)
    handler = server.server.call_tool_handler
    assert handler is not None
    result = asyncio.run(handler("search_api", {"query": "login"}))
    assert len(result) == 1


def test_authenticate_error_does_not_reveal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(monkeypatch)
    handler = server.server.call_tool_handler
    assert handler is not None
    with pytest.raises(AuthenticationError) as excinfo:
        asyncio.run(
            handler("search_api", {"query": "login", ACCESS_TOKEN_FIELD: "wrong-token"})
        )
    assert VALID_TOKEN not in str(excinfo.value)
    assert "***" not in str(excinfo.value)
