import argparse
import yaml
import requests
import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any
from caracal_rag.config import Config
from caracal_rag.sources import SourceFetcher
from caracal_rag.chunking import Chunker
from caracal_rag.embeddings import EmbeddingsGenerator
from caracal_rag.vectorstore import VectorStore
from caracal_rag.indexing import IndexingPipeline
from caracal_rag.mcp import MCPServer


def main():
    parser = argparse.ArgumentParser(prog="caracal-rag", description="Caracal RAG - retrieval-augmented generation toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser("check", help="Validate configuration and connectivity")
    check_parser.add_argument("--config", default="config/sources.example.yaml", help="Path to configuration file")

    # index command
    index_parser = subparsers.add_parser("index", help="Run the indexing pipeline")
    index_parser.add_argument("--config", default="config/sources.example.yaml", help="Path to configuration file")
    index_parser.add_argument("--source", help="Index only a specific source")

    # mcp command
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument("--config", default="config/sources.example.yaml", help="Path to configuration file")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port to run the MCP server on")

    args = parser.parse_args()

    if args.command == "check":
        check_config(args.config)
    elif args.command == "index":
        run_index(args.config, args.source)
    elif args.command == "mcp":
        run_mcp_server(args.config, args.port)
    else:
        parser.print_help()


def check_config(config_path: str):
    print("🔍 Validating configuration...")

    try:
        config = Config.load(config_path)
        print("✅ Configuration loaded successfully")

        # Test LiteLLM connectivity
        if config.embedding.api_base:
            print(f"📡 Testing LiteLLM at {config.embedding.api_base}...")
            response = requests.post(
                f"{config.embedding.api_base}/models",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ LiteLLM connectivity verified")
            else:
                print(f"❌ LiteLLM connectivity failed: {response.status_code}")

        # Test Chroma connectivity
        if config.chroma.host:
            print(f"🗄️ Testing Chroma at {config.chroma.host}:{config.chroma.port}...")
            # Try to list collections
            response = requests.get(
                f"http://{config.chroma.host}:{config.chroma.port}/api/v1/collections",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Chroma connectivity verified")
            else:
                print(f"❌ Chroma connectivity failed: {response.status_code}")

        print("\nConfiguration summary:")
        print(f"  - Embedding model: {config.embedding.model}")
        print(f"  - Embedding API: {config.embedding.api_base}")
        print(f"  - Chroma collection: {config.chroma.collection}")
        print(f"  - Sources configured: {len(config.sources)}")
        print("\n✅ Configuration validation complete")

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


def run_index(config_path: str, source_name: str = None):
    print("🔄 Starting indexing pipeline...")

    try:
        config = Config.load(config_path)
        pipeline = IndexingPipeline(config)

        if source_name:
            print(f"📋 Indexing only source: {source_name}")
            pipeline.run_source(source_name)
        else:
            print("📋 Indexing all configured sources...")
            pipeline.run_all_sources()

        print("✅ Indexing pipeline completed successfully")

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        raise


def run_mcp_server(config_path: str, port: int):
    print(f"🚀 Starting MCP server on port {port}...")

    try:
        config = Config.load(config_path)
        server = MCPServer(config, port)
        server.run()

    except Exception as e:
        print(f"❌ MCP server failed: {e}")
        raise


if __name__ == "__main__":
    main()