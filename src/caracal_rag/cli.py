from __future__ import annotations

import argparse
import sys

from caracal_rag.config import AppConfig
from caracal_rag.indexing import Indexer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caracal_rag")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--source", default=None, help="Index only the named source")

    subparsers.add_parser("check")

    return parser


def run_index(source: str | None) -> None:
    config = AppConfig.from_yaml("config/sources.example.yaml")
    indexer = Indexer(config=config, source_filter=source)
    indexer.run()


def run_check() -> None:
    config = AppConfig.from_yaml("config/sources.example.yaml")
    print(f"sources={len(config.sources)}")
    print(f"embedding_api_base={config.embedding.api_base}")
    print(f"chroma={config.chroma.host}:{config.chroma.port}")
    print("ok")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        run_index(args.source)
    elif args.command == "check":
        run_check()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
