from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from caracal_rag.auth import SimpleTokenVerifier
from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.vectorstore import (
    _chroma_client,
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    ensure_collection,
    search_chunks,
)

#: Name of the protected interface. The token is NEVER a tool argument: it is
#: supplied out-of-band through the HTTP ``Authorization`` header on the
#: authenticated transport, so it never reaches a model-visible tool schema,
#: tool argument, tool result, or LLM-facing transcript.
PROTECTED_INTERFACE_NAME = "caracal-rag"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


class SearchInputError(ValueError):
    """Validation failure for untrusted MCP search arguments."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_search_arguments(query: Any, top_k: Any) -> tuple[str, int]:
    """Validate search arguments strictly without coercion."""
    if not isinstance(query, str):
        raise SearchInputError("invalid_query", "query must be a string.")
    query = query.strip()
    if not query:
        raise SearchInputError(
            "invalid_query", "query must contain at least one non-whitespace character."
        )
    # bool is an int subclass, so require the exact int type.
    if type(top_k) is not int:
        raise SearchInputError(
            "invalid_top_k", f"top_k must be an integer between 1 and {MAX_TOP_K}."
        )
    if not 1 <= top_k <= MAX_TOP_K:
        raise SearchInputError(
            "invalid_top_k", f"top_k must be between 1 and {MAX_TOP_K} inclusive."
        )
    return query, top_k


def _tool_error_text(code: str, message: str) -> str:
    """Serialize a stable structured error for MCP clients."""
    return json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False)


@dataclass
class SearchResult:
    source: str
    document: str
    url: str
    content: str


class CaracalMcpServer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_model = config.embedding.model
        self.client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        self.collection = ensure_collection(self.client, config.chroma.collection)
        self.server = Server(PROTECTED_INTERFACE_NAME)
        self._register_handlers(self.server)

    def _register_handlers(self, server):
        # The SDK 1.x ``Server`` exposes ``list_tools``/``call_tool`` as
        # decorators; 2.x removed them in favour of ``add_request_handler``.
        list_tools_attr = getattr(server, "list_tools", None)
        call_tool_attr = getattr(server, "call_tool", None)
        if callable(list_tools_attr) and callable(call_tool_attr):
            @list_tools_attr()
            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name="search_api",
                        description=(
                            "Semantic search over indexed documentation chunks. "
                            "Protected: every call must be authenticated through the "
                            "interface's out-of-band access-token channel."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Non-empty search query.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": MAX_TOP_K,
                                    "description": f"Number of results (1-{MAX_TOP_K}).",
                                    "default": DEFAULT_TOP_K,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                ]

            @call_tool_attr()
            async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
                if name == "search_api":
                    query = arguments.get("query")
                    top_k = int(arguments.get("top_k", 5))
                    results = self.search(query=query, top_k=top_k)
                    return [TextContent(type="text", text=self._format(results))]
                raise ValueError(f"Unknown tool: {name}")
        else:
            # mcp 2.x: register request handlers via add_request_handler. The
            # result/request param models live in ``mcp.types`` (NOT
            # ``mcp.server.types``, which does not exist in any shipped SDK).
            from mcp.types import (
                CallToolRequestParams,
                CallToolResult,
                ListToolsResult,
                PaginatedRequestParams,
            )

            async def list_tools_handler(
                context, params: PaginatedRequestParams
            ) -> ListToolsResult:
                return ListToolsResult(tools=[
                    Tool(
                        name="search_api",
                        description=(
                            "Semantic search over indexed documentation chunks. "
                            "Protected: every call must be authenticated through the "
                            "interface's out-of-band access-token channel."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Non-empty search query.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": MAX_TOP_K,
                                    "description": f"Number of results (1-{MAX_TOP_K}).",
                                    "default": DEFAULT_TOP_K,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                ])

            async def call_tool_handler(
                context, params: CallToolRequestParams
            ) -> CallToolResult:
                name = params.name
                arguments = params.arguments or {}
                if name == "search_api":
                    query = arguments.get("query")
                    top_k = int(arguments.get("top_k", 5))
                    results = self.search(query=query, top_k=top_k)
                    return CallToolResult(
                        content=[TextContent(type="text", text=self._format(results))],
                        is_error=False,
                    )
                raise ValueError(f"Unknown tool: {name}")

            server.add_request_handler("tools/list", PaginatedRequestParams, list_tools_handler)
            server.add_request_handler("tools/call", CallToolRequestParams, call_tool_handler)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        assert_embedding_model_consistent(self.collection, self.embedding_model)
        embedding = embed_texts(
            [query],
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
            model=self.embedding_model,
        )[0]
        assert_embedding_dim_consistent(self.collection, len(embedding))
        response = search_chunks(self.collection, embedding, top_k=top_k)
        return self._parse_results(response)

    @staticmethod
    def _parse_results(response: dict[str, Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        if documents and isinstance(documents[0], list):
            documents = [item for sublist in documents for item in sublist]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = [item for sublist in metadatas for item in sublist]
        for doc_text, metadata in zip(documents, metadatas):
            if isinstance(doc_text, list):
                doc_text = doc_text[0] if doc_text else ""
            if isinstance(metadata, list):
                metadata = metadata[0] if metadata else {}
            results.append(
                SearchResult(
                    source=metadata.get("source", ""),
                    document=metadata.get("document", ""),
                    url=metadata.get("url", ""),
                    content=doc_text,
                )
            )
        return results

    def _format(self, results: list[SearchResult]) -> str:
        lines = []
        for item in results:
            lines.append(f"Source: {item.source}")
            lines.append(f"Document: {item.document}")
            lines.append(f"URL: {item.url}")
            lines.append("Content:")
            lines.append(item.content)
            lines.append("---")
        return "\n".join(lines)

    def http_app(self):
        """Return the Starlette ASGI app for the authenticated HTTP transport.

        The SDK's StreamableHTTP transport is wrapped with the SDK's
        ``BearerAuthMiddleware`` and a :class:`SimpleTokenVerifier`, so every
        request must present a valid ``Authorization: Bearer <token>`` header.
        The token is consumed from the HTTP request, never from a tool
        argument, so the secret never appears in the model-visible tool schema,
        tool arguments, tool results, or LLM-facing transcript.
        """
        from mcp.server.auth.settings import AuthSettings
        from mcp.server.transport_security import TransportSecuritySettings

        host = self.config.mcp.host
        port = self.config.mcp.port
        path = self.config.mcp.path
        if host in {"127.0.0.1", "localhost", "::1"}:
            transport_security = None
        else:
            transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[host, f"{host}:*"],
                allowed_origins=[f"http://{host}", f"http://{host}:*"],
            )
        auth_kwargs = dict(
            issuer_url=f"http://{host}:{port}",
            resource_server_url=f"http://{host}:{port}{path}",
        )
        # validate_token_resource was added in mcp 2.x; 1.x silently ignores
        # unknown kwargs only if extra='ignore' (it is not), so gate it.
        if "validate_token_resource" in AuthSettings.model_fields:
            auth_kwargs["validate_token_resource"] = True
        auth_settings = AuthSettings(**auth_kwargs)
        token_verifier = SimpleTokenVerifier(
            self.config.access_token,
            resource=auth_settings.resource_server_url,
        )

        # 1.x ships ``FastMCP`` (which owns its own low-level Server and is the
        # only way to mount the bearer auth middleware there); 2.x removed
        # FastMCP and the low-level ``Server`` itself accepts the auth kwargs.
        try:
            from mcp.server.fastmcp import FastMCP
        except Exception:
            return self.server.streamable_http_app(
                streamable_http_path=path,
                host=host,
                transport_security=transport_security,
                auth=auth_settings,
                token_verifier=token_verifier,
            )

        fastmcp = FastMCP(
            name=PROTECTED_INTERFACE_NAME,
            token_verifier=token_verifier,
            auth=auth_settings,
            host=host,
            port=port,
            transport_security=transport_security,
        )
        fastmcp._mcp_server = self.server  # noqa: SLF001
        return fastmcp.streamable_http_app()

    async def run_http_async(self) -> None:
        import uvicorn

        app = self.http_app()
        config = uvicorn.Config(
            app,
            host=self.config.mcp.host,
            port=self.config.mcp.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_stdio_async(self) -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
