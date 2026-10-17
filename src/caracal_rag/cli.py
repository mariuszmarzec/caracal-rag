from __future__ import annotations

import argparse
import asyncio
import sys

from caracal_rag.config import AppConfig
from caracal_rag.embeddings import embed_texts
from caracal_rag.indexing import Indexer
from caracal_rag.vectorstore import _chroma_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caracal_rag")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--source", default=None, help="Index only the named source")

    subparsers.add_parser("check")

    mcp_parser = subparsers.add_parser("mcp")
    mcp_parser.add_argument(
        "--host",
        default=None,
        help="MCP server host (unused; stdio transport)",
    )
    mcp_parser.add_argument(
        "--port",
        default=None,
        help="MCP server port (unused; stdio transport)",
    )

    return parser

def run_index(source: str | None) -> None:
    config = AppConfig.from_yaml("config/sources.example.yaml")
    indexer = Indexer(config=config, source_filter=source)
    indexer.run()

def run_check(config_path: str = "config/sources.example.yaml") -> None:
    """Validate the configuration and connectivity to LiteLLM and Chroma.

    Unlike a pure config parse, this actually connects to the configured
    endpoints: it asks Chroma for a heartbeat and issues a probe embedding
    through LiteLLM, so a misconfigured or unreachable service fails fast
    with a non-zero exit code.
    """
    try:
        config = AppConfig.from_yaml(config_path)
        print(f"sources={len(config.sources)}")
        print(f"embedding_api_base={config.embedding.api_base}")
        print(f"chroma={config.chroma.host}:{config.chroma.port}")

        client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        heartbeat = client.heartbeat()
        print(f"chroma_heartbeat={heartbeat}")

        vector = embed_texts(
            ["caracal-rag check"],
            api_base=config.embedding.api_base,
            api_key=config.embedding.api_key,
            model=config.embedding.model,
        )
        print(f"embedding_dim={len(vector[0])}")
        print("ok")
    except Exception as exc:  # noqa: BLE001 - surface any connectivity failure
        print(f"check failed: {exc}", file=sys.stderr)
        sys.exit(1)

def run_mcp(host_override: str | None = None, port_override: str | None = None) -> None:
    from caracal_rag.mcp import CaracalMcpServer

    config = AppConfig.from_yaml("config/sources.example.yaml")

    # Allow host/port overrides via CLI for connection debugging
    if host_override is not None:
        config.chroma.host = host_override
    if port_override is not None:
        config.chroma.port = int(port_override)

    server = CaracalMcpServer(config)

    async def _run() -> None:
        await server.server.run_stdio_async()

    asyncio.run(_run())

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        run_index(args.source)
    elif args.command == "check":
        run_check()
    elif args.command == "mcp":
        run_mcp(args.host, args.port)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
