"""Tests for the MCP tools: search_api.

External services (LiteLLM, Chroma) are replaced with in-memory fakes so the
tests run without network access.
"""

from __future__ import annotations

import asyncio
import json
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
    # The raw token is no longer a tool argument: it is supplied out-of-band
    # through the HTTP transport, so the low-level handlers never see it.
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


def _call_search_tool(server: CaracalMcpServer, arguments: dict[str, Any]):
    handler = server.server.call_tool_handler
    assert handler is not None
    return asyncio.run(handler("search_api", arguments))


def _parse_tool_error(result) -> dict[str, Any]:
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert "error" in payload
    return payload["error"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": None},
        {"query": 123},
        {"query": ""},
        {"query": "   \n\t  "},
    ],
)
def test_call_tool_rejects_invalid_queries_without_backend_calls(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    server = _make_server(monkeypatch)
    embed_calls: list[Any] = []
    monkeypatch.setattr(
        "caracal_rag.mcp.embed_texts",
        lambda texts, **kwargs: embed_calls.append(texts) or [[0.1, 0.2]],
    )

    result = _call_search_tool(server, arguments)

    error = _parse_tool_error(result)
    assert error["code"] == "invalid_query"
    assert error["message"]
    assert embed_calls == []


@pytest.mark.parametrize("top_k", [True, False, 2.5, "5", None, 0, -1, 21])
def test_call_tool_rejects_invalid_top_k_without_backend_calls(
    monkeypatch: pytest.MonkeyPatch, top_k: Any
) -> None:
    server = _make_server(monkeypatch)
    embed_calls: list[Any] = []
    monkeypatch.setattr(
        "caracal_rag.mcp.embed_texts",
        lambda texts, **kwargs: embed_calls.append(texts) or [[0.1, 0.2]],
    )

    result = _call_search_tool(server, {"query": "login", "top_k": top_k})

    error = _parse_tool_error(result)
    assert error["code"] == "invalid_top_k"
    assert "1 and 20" in error["message"]
    assert embed_calls == []


@pytest.mark.parametrize("top_k", [1, 20])
def test_call_tool_accepts_top_k_bounds_and_trims_query(
    monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    server = _make_server(monkeypatch)
    embedded_queries: list[list[str]] = []
    searched_top_k: list[int] = []
    monkeypatch.setattr(
        "caracal_rag.mcp.embed_texts",
        lambda texts, **kwargs: embedded_queries.append(list(texts)) or [[0.1, 0.2]],
    )
    monkeypatch.setattr(
        "caracal_rag.mcp.search_chunks",
        lambda collection, embedding, top_k=5: (
            searched_top_k.append(top_k)
            or {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        ),
    )

    result = _call_search_tool(server, {"query": "  login \n", "top_k": top_k})

    assert len(result) == 1
    assert result[0].text == ""  # Fake search returns no results at either boundary.
    assert embedded_queries == [["login"]]
    assert searched_top_k == [top_k]


@pytest.mark.parametrize("backend", ["embedding", "chroma"])
def test_backend_failure_is_sanitized_and_following_request_succeeds(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    from caracal_rag.vectorstore import search_chunks as real_search_chunks

    server = _make_server(monkeypatch)
    calls = 0
    sensitive_details = "Bearer private-test-token api_key=private-test-key"

    if backend == "embedding":
        def flaky_embedding(texts, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(f"backend request failed: {sensitive_details}")
            return [[0.1, 0.2]]

        monkeypatch.setattr("caracal_rag.mcp.embed_texts", flaky_embedding)
    else:
        def flaky_search(collection, embedding, top_k=5):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(f"backend request failed: {sensitive_details}")
            return real_search_chunks(collection, embedding, top_k=top_k)

        monkeypatch.setattr("caracal_rag.mcp.search_chunks", flaky_search)

    failed = _call_search_tool(server, {"query": "login"})
    error = _parse_tool_error(failed)
    failed_text = failed[0].text
    assert error["code"] == "search_failed"
    assert "caracal-rag check" in error["message"]
    assert "private-test-token" not in failed_text
    assert "private-test-key" not in failed_text
    assert sensitive_details not in failed_text

    recovered = _call_search_tool(server, {"query": "login"})
    assert len(recovered) == 1
    assert "Source: fiteo-docs" in recovered[0].text
    assert calls == 2
