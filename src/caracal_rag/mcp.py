from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.vectorstore import _chroma_client, ensure_collection, search_chunks


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
                            "query": {"type": "string", "description": "Search query."},
                            "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
                        },
                        "required": ["query"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name != "search_api":
                raise ValueError(f"Unknown tool: {name}")
            query = arguments.get("query")
            top_k = int(arguments.get("top_k", 5))
            results = self.search(query=query, top_k=top_k)
            return [TextContent(type="text", text=self._format(results))]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedding = embed_texts(
            [query],
            model=self.config.embedding.model,
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
        )[0]
        response = search_chunks(self.collection, embedding, top_k=top_k)
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
