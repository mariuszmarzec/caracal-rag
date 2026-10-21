"""Tests for the authenticated HTTP MCP transport.

These tests exercise the real MCP SDK end-to-end: the StreamableHTTP app
built by ``CaracalMcpServer.http_app()`` must enforce the out-of-band
``Authorization: Bearer <token>`` channel on every request, must reject
missing or invalid tokens with a 401, and must never expose the raw access
token in the model-visible tool schema, tool arguments, or tool results.

The other MCP test modules stub the SDK in ``sys.modules`` at import time, so
this module restores the real ``mcp`` package and reloads ``caracal_rag.mcp``
before use; chromadb and litellm remain stubbed so the tests stay offline.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from typing import Any

import pytest

# chromadb and litellm stay stubbed so the tests run offline; the real MCP
# SDK is restored instead of stubbed so the HTTP auth boundary is exercised
# against the actual SDK code.
chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

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

# The other MCP test modules stub ``mcp`` in ``sys.modules`` at import time, so
# this module must restore the real SDK before use. That cannot be done once at
# module level: pytest imports every test module during collection (alphabetically),
# so a later-collected module can re-stub ``sys.modules["mcp.types"]`` before the
# session actually runs. Restoring in an autouse fixture guarantees the real SDK
# is present for every test in this module regardless of collection order.
_MCP_SUBMODULES = [
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp",
    "mcp.server.auth",
    "mcp.server.auth.provider",
    "mcp.server.auth.settings",
    "mcp.server.auth.middleware",
    "mcp.server.auth.middleware.bearer_auth",
    "mcp.server.auth.routes",
    "mcp.server.transport_security",
    "mcp.server.streamable_http",
    "mcp.server.streamable_http_manager",
    "mcp.types",
    "mcp.shared.message",
    "mcp.shared.session",
]

# Restore at module level too: a previously collected module may have stubbed
# ``mcp`` as a plain module, which makes ``import caracal_rag.mcp`` fail at
# collection time (its top-level ``from mcp.server import Server`` needs the
# real package). The fixture below re-restores before every test execution.
for name in _MCP_SUBMODULES:
    sys.modules.pop(name, None)
import mcp  # noqa: F401,E402  (real package)

import caracal_rag.mcp as mcp_module  # noqa: E402

mcp_module = importlib.reload(mcp_module)


@pytest.fixture(autouse=True)
def _restore_real_mcp() -> None:
    """Restore the real MCP SDK before each test.

    Only ``sys.modules`` is restored, never ``caracal_rag.mcp``: reloading that
    module would re-execute its class definitions and break ``isinstance`` checks
    in other test modules that bound those classes at collection time. The
    module's own top-level ``from mcp.server import Server`` etc. were bound when
    this module collected (with the real SDK), so they stay valid; this fixture
    just makes sure the lazy ``from mcp.types import ...`` inside
    ``_register_handlers`` resolves to the real SDK at execution time.
    """
    for name in _MCP_SUBMODULES:
        sys.modules.pop(name, None)
import mcp  # noqa: F401,F811,E402  (real package; F811: re-import in fixture)

# --- minimal ASGI test client (no heavy dependency needed) -------------------
ACCEPT = "application/json, text/event-stream"


def _parse_sse(body: str) -> dict[str, Any] | None:
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


class _McpClient:
    """Drive the StreamableHTTP MCP app in-process via Starlette's TestClient.

    The TestClient runs the ASGI lifespan (so the SDK's session-manager task
    group is initialized) and handles the MCP session-id header automatically.
    """

    def __init__(self, app: Any, *, auth: str | None = None) -> None:
        from starlette.testclient import TestClient

        self._client = TestClient(app)
        self._headers: dict[str, str] = {"accept": ACCEPT}
        if auth is not None:
            self._headers["authorization"] = auth

    def __enter__(self) -> "_McpClient":
        # Entering the TestClient runs the ASGI lifespan, which starts the
        # SDK's session-manager task group (required before any request).
        self._client.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.__exit__(*exc)

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
        response = self._client.post(path, json=payload, headers=self._headers)
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._headers["mcp-session-id"] = session_id
        return response.status_code, dict(response.headers), response.text


TOKEN = "correct-horse-battery-staple"


class _FakeChroma:
    def get_or_create_collection(self, name: str) -> Any:
        class _Collection:
            def get(self, where=None, include=None):
                return {"ids": [], "metadatas": []}

            def query(self, query_embeddings, n_results: int = 5) -> dict[str, Any]:
                return {
                    "documents": [["Login details."]],
                    "metadatas": [[{"source": "api", "document": "login.md", "url": "https://example.com/login.md"}]],
                }

        return _Collection()


def _make_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str | None = TOKEN,
    enabled: bool = True,
) -> Any:
    monkeypatch.setattr(mcp_module, "_chroma_client", lambda *args, **kwargs: _FakeChroma())
    monkeypatch.setattr(mcp_module, "embed_texts", lambda texts, **kwargs: [[0.1, 0.2]])
    from caracal_rag.config import AppConfig

    config = AppConfig.from_yaml("config/sources.example.yaml")
    config.access_token.enabled = enabled
    config.access_token.token = token
    config.mcp.transport = "http"
    config.mcp.host = "testserver"
    config.mcp.port = 8000
    config.mcp.path = "/mcp"
    server = mcp_module.CaracalMcpServer(config)
    return server.http_app()


def test_http_rejects_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch)
    with _McpClient(app) as client:
        status, _, _ = client.post(
            "/mcp",
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            }},
        )
    assert status == 401


def test_http_rejects_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch)
    client = _McpClient(app)
    client._headers["authorization"] = "Bearer wrong-token"
    status, _, _ = client.post(
        "/mcp",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "1"},
        }},
    )
    assert status == 401


def test_http_valid_bearer_hides_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid Bearer token authenticates; the token itself must never appear
    in the model-visible tool schema or in the tool call result."""
    app = _make_app(monkeypatch)
    with _McpClient(app, auth=f"Bearer {TOKEN}") as client:
        status, headers, body = client.post(
            "/mcp",
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            }},
        )
        assert status == 200, body
        session = headers.get("mcp-session-id")
        assert session, "initialize must return an mcp-session-id"

        status, _, body = client.post("/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert status == 200, body
        tools = _parse_sse(body)["result"]["tools"]

        status, _, body = client.post(
            "/mcp",
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "search_api", "arguments": {"query": "login"}}},
        )
        assert status == 200, body
        result = _parse_sse(body)["result"]
        text = result["content"][0]["text"]

    assert [t["name"] for t in tools] == ["search_api"]
    schema = tools[0]["inputSchema"]
    tools_blob = json.dumps(tools)
    assert "access_token" not in tools_blob, (
        "raw token field leaked into the tool schema"
    )
    assert "access_token" not in schema["properties"]
    assert "access_token" not in schema.get("required", [])
    assert TOKEN not in tools_blob
    assert TOKEN not in text
    assert "Source: api" in text
    assert "Document: login.md" in text


def test_simple_token_verifier_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit-check the verifier against the SDK protocol surface."""
    from mcp.server.auth.provider import AccessToken

    from caracal_rag.auth import SimpleTokenVerifier
    from caracal_rag.config import AccessTokenConfig

    async def run() -> None:
        verifier = SimpleTokenVerifier(AccessTokenConfig(enabled=True, token=TOKEN))
        result = await verifier.verify_token(TOKEN)
        assert isinstance(result, AccessToken)
        assert result.token == TOKEN
        assert await verifier.verify_token("wrong-token") is None

        disabled = SimpleTokenVerifier(AccessTokenConfig(enabled=False, token=None))
        result = await disabled.verify_token("anything-at-all")
        assert isinstance(result, AccessToken)

        # A misconfigured (too short) configured token must not authenticate
        # anything, even when the presented token matches it verbatim.
        misconfigured = SimpleTokenVerifier(AccessTokenConfig(enabled=True, token="short"))
        assert await misconfigured.verify_token("short") is None

    asyncio.run(run())


def test_http_handled_search_errors_are_sanitized_and_session_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled invalid/backend query returns a tool error; the session stays usable."""
    app = _make_app(monkeypatch)
    embed_calls = 0
    private_details = "Bearer very-private-token api_key=private-key-value"

    def fail_first_embedding(texts, **kwargs):
        nonlocal embed_calls
        embed_calls += 1
        if embed_calls == 1:
            raise RuntimeError(f"request failed: {private_details}")
        return [[0.1, 0.2]]

    monkeypatch.setattr(mcp_module, "embed_texts", fail_first_embedding)
    with _McpClient(app, auth=f"Bearer {TOKEN}") as client:
        status, _, body = client.post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert status == 200, body

        status, _, body = client.post(
            "/mcp",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        assert status == 200, body
        tools = _parse_sse(body)["result"]["tools"]
        schema = tools[0]["inputSchema"]
        assert schema["properties"]["query"]["minLength"] == 1
        assert schema["properties"]["top_k"]["minimum"] == 1
        assert schema["properties"]["top_k"]["maximum"] == 20

        status, _, body = client.post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search_api",
                    "arguments": {"query": "   \n\t"},
                },
            },
        )
        assert status == 200, body
        invalid_result = _parse_sse(body)["result"]
        assert invalid_result["isError"] is True
        invalid_text = invalid_result["content"][0]["text"]
        invalid_error = json.loads(invalid_text)["error"]
        assert invalid_error["code"] == "invalid_query"
        assert TOKEN not in invalid_text

        status, _, body = client.post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "search_api",
                    "arguments": {"query": "login", "top_k": 5},
                },
            },
        )
        assert status == 200, body
        backend_result = _parse_sse(body)["result"]
        assert backend_result["isError"] is True
        backend_text = backend_result["content"][0]["text"]
        backend_error = json.loads(backend_text)["error"]
        assert backend_error["code"] == "search_failed"
        assert "caracal-rag check" in backend_error["message"]
        assert "very-private-token" not in backend_text
        assert "private-key-value" not in backend_text
        assert private_details not in backend_text
        assert TOKEN not in backend_text

        # A handled failure must not poison the running MCP session.
        status, _, body = client.post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "search_api",
                    "arguments": {"query": "login", "top_k": 20},
                },
            },
        )
        assert status == 200, body
        recovered_result = _parse_sse(body)["result"]
        assert recovered_result.get("isError", False) is False
        recovered_text = recovered_result["content"][0]["text"]
        assert "Source: api" in recovered_text
        assert "Document: login.md" in recovered_text
        assert TOKEN not in recovered_text

    assert embed_calls == 2
