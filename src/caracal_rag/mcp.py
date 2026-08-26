import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from mcp import Server, Tool
from mcp.types import TextContent
from caracal_rag.config import Config
from caracal_rag.embeddings import EmbeddingsGenerator
from caracal_rag.vectorstore import VectorStore


class MCPServer:
    """MCP server exposing search_api for semantic search."""

    def __init__(self, config: Config, port: int = 8000):
        self.config = config
        self.port = port
        self.embeddings = EmbeddingsGenerator(config.embedding)
        self.vectorstore = VectorStore(config.chroma)
        self.server = Server("caracal-rag")
        self._setup_tools()

    def _setup_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_api",
                    description="Search indexed knowledge base for relevant documentation chunks",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "search_api":
                return await self._search_api(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_api(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Perform semantic search against Chroma."""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)

        if not query:
            return [TextContent(type="text", text="Error: Empty query")]

        try:
            # Generate embedding for query
            query_embedding = self.embeddings.generate(query)

            # Search Chroma
            results = self.vectorstore.query_similar(query_embedding, limit=limit)

            # Format results
            formatted_results = self._format_results(results)
            return [TextContent(type="text", text=formatted_results)]

        except Exception as e:
            return [TextContent(type="text", text=f"Search error: {str(e)}")]

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format search results for display."""
        if not results or "documents" not in results:
            return "No results found."

        output = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
            distance = results.get("distances", [[]])[0][i] if results.get("distances") else 0

            source = metadata.get("source", "unknown")
            document = metadata.get("document", "unknown")
            url = metadata.get("url", "unknown")
            chunk_type = metadata.get("type", "unknown")

            output.append(f"Result {i + 1} (similarity: {1 - distance:.3f}):")
            output.append(f"  Source: {source}")
            output.append(f"  Document: {document}")
            output.append(f"  URL: {url}")
            output.append(f"  Type: {chunk_type}")
            output.append(f"  Content:\n{doc}")
            output.append("")

        return "\n".join(output)

    def run(self):
        """Run the MCP server over stdio."""
        import asyncio
        from mcp.server.stdio import stdio_server

        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

        asyncio.run(main())