from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.vectorstore import (
    _chroma_client,
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    ensure_collection,
    search_chunks,
)


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
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "search_api":
                query = arguments.get("query")
                top_k = int(arguments.get("top_k", 5))
                results = self.search(query=query, top_k=top_k)
                return [TextContent(type="text", text=self._format(results))]
            raise ValueError(f"Unknown tool: {name}")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # Never mix query vectors from the current model with stored vectors
        # from a different embedding space -- the similarity scores would be
        # meaningless. Fail with actionable rebuild guidance instead.
        assert_embedding_model_consistent(self.collection, self.embedding_model)
        embedding = embed_texts(
            [query],
            api_base=self.config.embedding.api_base,
            api_key=self.config.embedding.api_key,
            model=self.embedding_model,
        )[0]
        # The query vector's dimension is checked against the stored vectors too:
        # two models that happen to agree on family but differ in dimensionality
        # still produce incompatible spaces, and similarity scores across them
        # are meaningless.
        assert_embedding_dim_consistent(self.collection, len(embedding))
        response = search_chunks(self.collection, embedding, top_k=top_k)
        return self._parse_results(response)

    @staticmethod
    def _parse_results(response: dict[str, Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []

        # Chroma nests the top-k results: documents=[[...top-k texts...]],
        # metadatas=[[...top-k metadata...]]. Flatten the outer array so every
        # result is returned instead of only the first nested item.
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
