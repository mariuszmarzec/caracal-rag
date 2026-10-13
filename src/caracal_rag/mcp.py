from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.endpoints import (
    endpoint_method,
    extract_endpoint,
    normalize_path,
    path_only,
)
from caracal_rag.vectorstore import _chroma_client, ensure_collection, search_chunks

# Cap how many candidate chunks we pull from Chroma before endpoint filtering.
_ENDPOINT_CANDIDATE_K = 25


@dataclass
class SearchResult:
    source: str
    document: str
    url: str
    content: str


class CaracalMcpServer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        self.collection = ensure_collection(self.client, config.chroma.collection)
        self.server = Server("caracal-rag")

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="search_api",
                    description="Semantic search over indexed documentation chunks.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results.",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="find_endpoint",
                    description=(
                        "Look up indexed API documentation for an exact HTTP endpoint "
                        "path (e.g. '/fiteo/api/1/tasks' or 'GET /fiteo/api/1/tasks'). "
                        "Performs a normalised lookup against the indexed API docs and "
                        "returns the matching document/chunks with source metadata. "
                        "Reports clearly when no endpoint is found."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Endpoint path, optionally prefixed with the HTTP "
                                    "method, e.g. '/fiteo/api/1/tasks' or "
                                    "'POST /fiteo/api/1/login'."
                                ),
                            },
                            "method": {
                                "type": "string",
                                "description": (
                                    "Optional HTTP method to constrain the lookup "
                                    "(GET, POST, PUT, DELETE, ...)."
                                ),
                            },
                        },
                        "required": ["path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "search_api":
                query = arguments.get("query")
                top_k = int(arguments.get("top_k", 5))
                results = self.search(query=query, top_k=top_k)
                return [TextContent(type="text", text=self._format(results))]
            if name == "find_endpoint":
                path = arguments.get("path")
                method = arguments.get("method")
                results = self.find_endpoint(path=path, method=method)
                return [TextContent(type="text", text=self._format_endpoint(results, path, method))]
            raise ValueError(f"Unknown tool: {name}")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedding = embed_texts(
            [query],
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
        )[0]
        response = search_chunks(self.collection, embedding, top_k=top_k)
        return self._parse_results(response)

    def find_endpoint(self, path: str, method: str | None = None) -> list[SearchResult]:
        """Look up indexed API documentation for an exact endpoint path.

        Returns matching chunks with source metadata. The lookup is normalised
        so method prefix, trailing slashes and repeated slashes do not matter.
        Returns an empty list when no indexed chunk declares the requested path.
        """
        if not path:
            return []

        target_path = path_only(path)
        target_method = (method or endpoint_method(path) or "").upper() or None

        # Embed the bare path so Chroma returns candidate chunks from the API docs.
        embedding = embed_texts(
            [target_path],
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
        )[0]
        response = search_chunks(self.collection, embedding, top_k=_ENDPOINT_CANDIDATE_K)
        candidates = self._parse_results(response)

        matches: list[SearchResult] = []
        for candidate in candidates:
            endpoint = extract_endpoint(candidate.content)
            if endpoint is None:
                continue
            if normalize_path(endpoint.path) != target_path:
                continue
            if target_method is not None and endpoint.method.upper() != target_method:
                continue
            matches.append(candidate)
        return matches

    @staticmethod
    def _parse_results(response: dict[str, Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
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

    def _format_endpoint(
        self,
        results: list[SearchResult],
        path: str | None,
        method: str | None,
    ) -> str:
        target_path = path_only(path or "")
        target_method = (method or endpoint_method(path or "") or "").upper() or None

        header = f"Endpoint: {target_method or 'ANY'} {target_path}".strip()
        if not results:
            return (
                f"{header}\n"
                "No match: no indexed API documentation declares this endpoint path."
            )

        lines = [header, f"Matches: {len(results)}"]
        for item in results:
            lines.append("---")
            lines.append(f"Source: {item.source}")
            lines.append(f"Document: {item.document}")
            lines.append(f"URL: {item.url}")
            lines.append("Content:")
            lines.append(item.content)
        return "\n".join(lines)
