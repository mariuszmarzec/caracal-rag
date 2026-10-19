from __future__ import annotations

import argparse
import asyncio
import os
import sys

from caracal_rag.config import (
    EMBEDDING_API_KEY_ENV,
    AppConfig,
    ChromaConfig,
    ConfigError,
    EmbeddingConfig,
    load_active_config,
    resolve_home,
    write_config,
    write_secret_api_key,
)
from caracal_rag.embeddings import probe_embedding_dim, redact_secret, validate_embedding_endpoint
from caracal_rag.indexing import Indexer
from caracal_rag.manifest import (
    assert_manifest_consistent,
    build_manifest,
    write_manifest,
)
from caracal_rag.vectorstore import (
    _chroma_client,
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    ensure_collection,
)

# Example sources shipped with the checkout. Only used as a *template* for
# `caracal-rag setup` when the repository can be located (i.e. running from a
# source checkout or the installed tree); the active installed configuration
# under ``~/.caracal`` is what every runtime command actually uses.
_CHECKOUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLE_CONFIG = os.path.join(_CHECKOUT_DIR, "config", "sources.example.yaml")


def _repo_root() -> str | None:
    """Locate the repository root that contains ``config/sources.example.yaml``.

    Returns ``None`` when the checkout cannot be found (for example, a
    `pip install`-ed wheel without the config directory). In that case the
    installer falls back to a built-in template.
    """
    if os.path.isfile(_EXAMPLE_CONFIG):
        return _CHECKOUT_DIR
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caracal-rag")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--source", default=None, help="Index only the named source")
    index_parser.add_argument(
        "--config",
        default=None,
        help="Path to the active configuration file (default: the installed configuration)",
    )

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument(
        "--config",
        default=None,
        help="Path to the active configuration file (default: the installed configuration)",
    )

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
    mcp_parser.add_argument(
        "--config",
        default=None,
        help="Path to the active configuration file (default: the installed configuration)",
    )

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument(
        "--model",
        default=None,
        help="FreeLLMAPI embedding family/model (for example bge-m3). "
        "Prompts interactively when omitted.",
    )
    setup_parser.add_argument(
        "--api-base",
        default=None,
        help="FreeLLMAPI base URL (default: http://localhost:3001/v1)",
    )
    return parser


def run_index(source: str | None, config_path: str | None = None) -> None:
    config = load_active_config(config_path)
    _ensure_api_key(config)
    # Probe the live dimension before touching the index. A mismatch against the
    # manifest or the collection is refused by _assert_index_safe, so a failed
    # probe never leaves the manifest in an inconsistent state.
    observed_dim = _probe_dim(config)
    _assert_index_safe(config, observed_dim=observed_dim)
    indexer = Indexer(config=config, source_filter=source, embedding_dim=observed_dim)
    indexer.run()
    _write_manifest(config, indexer=indexer, embedding_dim=observed_dim)


def _write_manifest(config: AppConfig, indexer=None, embedding_dim: int | None = None) -> None:
    home = resolve_home()
    if indexer is not None:
        # Read back the post-index committed state instead of using the indexer's
        # initial cache. The manifest must reflect the index that actually exists.
        collection_data = indexer.collection.get(include=["metadatas"])
        committed_urls = {
            (metadata or {}).get("url")
            for metadata in (collection_data.get("metadatas") or [])
            if not (metadata or {}).get("caracal_staging", False)
            and (metadata or {}).get("url")
        }
        document_count = len(committed_urls)
    else:
        document_count = 0
    # A failed manifest write is a failed operation: do not silently claim that
    # the index identity has been recorded when it has not.
    write_manifest(
        home.manifest_path,
        build_manifest(config, document_count, embedding_dim=embedding_dim),
    )


def run_check(config_path: str | None = None) -> None:
    """Validate the active configuration and connectivity to FreeLLMAPI and Chroma.

    Unlike a pure config parse, this actually connects to the configured
    endpoints: it asks Chroma for a heartbeat and issues a probe embedding
    through LiteLLM, so a misconfigured or unreachable service fails fast
    with a non-zero exit code.
    """
    config: AppConfig | None = None
    try:
        config = load_active_config(config_path)
        _ensure_api_key(config)
        # ``check`` must validate the *live* collection, not only the durable
        # manifest. A missing manifest does not make a pre-existing collection
        # compatible: vectors without model metadata (legacy/mixed) or tagged
        # with a different family or dimension are refused here, before any
        # probe call, so a mismatched index is never reported as healthy.
        observed_dim = _probe_dim(config)
        _assert_index_safe(config, observed_dim=observed_dim)
        print(f"config={config.config_path}")
        print(f"sources={len(config.sources)}")
        print(f"embedding_model={config.embedding.model}")
        print(f"embedding_api_base={config.embedding.api_base}")
        print(f"chroma={config.chroma.host}:{config.chroma.port}")

        client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        heartbeat = client.heartbeat()
        print(f"chroma_heartbeat={heartbeat}")

        print(f"embedding_dim={observed_dim}")
        print("ok")
    except Exception as exc:  # noqa: BLE001 - surface any connectivity failure
        secret = (
            config.embedding.api_key
            if config is not None
            else os.getenv(EMBEDDING_API_KEY_ENV)
        )
        print(redact_secret(f"check failed: {exc}", secret), file=sys.stderr)
        sys.exit(1)


def run_mcp(
    host_override: str | None = None,
    port_override: str | None = None,
    config_path: str | None = None,
) -> None:
    from caracal_rag.mcp import CaracalMcpServer

    config = load_active_config(config_path)
    _ensure_api_key(config)
    # Probe the live dimension before the MCP server starts serving. The
    # manifest and collection are validated against it, so a dimension drift
    # is refused before any query vector is generated.
    observed_dim = _probe_dim(config)
    _assert_index_safe(config, observed_dim=observed_dim)

    # Allow host/port overrides via CLI for connection debugging
    if host_override is not None:
        config.chroma.host = host_override
    if port_override is not None:
        config.chroma.port = int(port_override)

    server = CaracalMcpServer(config)

    async def _run() -> None:
        await server.server.run_stdio_async()

    asyncio.run(_run())


def _ensure_api_key(config: AppConfig) -> None:
    """Fail early with actionable guidance when the FreeLLMAPI key is absent.

    The key is never echoed: the message only names where to obtain it.
    """
    if not config.embedding.api_key:
        raise ConfigError(
            f"No FreeLLMAPI API key is configured. Provide one by running "
            "'caracal-rag setup', by exporting "
            f"{EMBEDDING_API_KEY_ENV}, or by setting `embedding.api_key` in the "
            "secrets file under your Caracal home (~/.caracal/config/secrets.yaml). "
            "The key must be a valid key for the configured FreeLLMAPI endpoint."
        )


def _assert_manifest_consistent(config: AppConfig, observed_dim: int | None = None) -> None:
    home = resolve_home()
    assert_manifest_consistent(home.manifest_path, config, observed_dim=observed_dim)


def _probe_dim(config: AppConfig) -> int:
    """Probe the configured embedding endpoint and return the vector dimension.

    Failures are wrapped in the same actionable ``ConfigError`` as
    ``validate_embedding_endpoint``, so a bad model or unreachable service
    aborts the runtime command with a non-zero exit instead of leaving the
    index in an unknown state.
    """
    from caracal_rag.config import ConfigError

    try:
        return probe_embedding_dim(
            api_base=config.embedding.api_base,
            model=config.embedding.model,
            api_key=config.embedding.api_key,
        )
    except Exception as exc:  # noqa: BLE001 - surface any probe failure
        raise ConfigError(
            f"Could not probe the embedding endpoint at "
            f"{config.embedding.api_base!r} for model {config.embedding.model!r}: "
            f"{redact_secret(str(exc), config.embedding.api_key)}. Verify the "
            "FreeLLMAPI service is running, the base URL is correct, and that "
            "model is enabled under FreeLLMAPI Models -> Embeddings."
        ) from exc


def _assert_index_safe(
    config: AppConfig, observed_dim: int | None = None
) -> None:
    """Pre-flight guard for indexing: manifest *and* the live collection.

    The manifest is the durable record under ``~/.caracal``; the collection is
    the live Chroma store. Both must agree with the active configuration. A
    missing manifest must not be read as "legacy vectors are compatible": if
    the collection already holds committed vectors, their embedding space is
    inspected directly and an unknown, legacy, mismatched or differently
    dimensional space is refused with rebuild guidance before any vector is
    written or reused.
    """
    _assert_manifest_consistent(config, observed_dim=observed_dim)
    client = _chroma_client(
        host=config.chroma.host,
        port=config.chroma.port,
        ssl=config.chroma.ssl,
    )
    collection = ensure_collection(client, config.chroma.collection)
    assert_embedding_model_consistent(collection, config.embedding.model)
    if observed_dim is not None:
        assert_embedding_dim_consistent(collection, observed_dim)


def run_setup(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    """Create or update the active installed configuration and secrets.

    Idempotent: existing configuration and secrets are preserved and only the
    fields the user explicitly provides are updated. The API key is read from
    a hidden prompt (not echoed) when not given on the command line.
    """
    home = resolve_home()
    interactive = sys.stdin.isatty()

    existing: AppConfig | None = None
    if home.config_path.exists():
        # Fail closed for invalid existing configuration. Treating a malformed
        # file as "not installed yet" would replace it with the example template.
        existing = load_active_config()

    resolved_api_base = (
        api_base
        or (existing.embedding.api_base if existing else "http://localhost:3001/v1")
        or "http://localhost:3001/v1"
    )
    resolved_model = model or (existing.embedding.model if existing else None)
    if not resolved_model:
        if interactive:
            resolved_model = input("Embedding family/model (e.g. bge-m3): ").strip() or None
        if not resolved_model:
            raise ConfigError(
                "No embedding model supplied. Re-run with --model <family> "
                "(check your FreeLLMAPI Models -> Embeddings page for what is "
                "available in your deployment)."
            )

    existing_key = existing.embedding.api_key if existing else None
    resolved_api_key = api_key or existing_key or os.getenv(EMBEDDING_API_KEY_ENV)
    if not resolved_api_key:
        if interactive:
            import getpass

            prompt = "FreeLLMAPI API key (input hidden): "
            resolved_api_key = getpass.getpass(prompt).strip() or None
        if not resolved_api_key:
            raise ConfigError(
                "No FreeLLMAPI API key supplied. Enter it at the hidden prompt, "
                f"or export {EMBEDDING_API_KEY_ENV} and re-run `caracal-rag setup`. "
                "Do not pass the key as a command-line argument: arguments can "
                "leak through shell history and process listings."
            )

    # Reuse the example config from the checkout when available so sources
    # carry over; otherwise start from an empty source list the user can edit.
    if existing is not None:
        config = existing
        config.embedding.model = resolved_model
        config.embedding.api_base = resolved_api_base
    elif _repo_root() is not None and os.path.isfile(_EXAMPLE_CONFIG):
        config = AppConfig.from_yaml(_EXAMPLE_CONFIG)
        config.embedding.model = resolved_model
        config.embedding.api_base = resolved_api_base
    else:
        config = AppConfig(
            sources=[],
            embedding=EmbeddingConfig(
                model=resolved_model,
                api_base=resolved_api_base,
                api_key=resolved_api_key,
            ),
            chroma=ChromaConfig(),
            config_path=home.config_path,
        )

    # Persist the active configuration (no secrets) and the secrets file.
    write_config(config, home.config_path)
    write_secret_api_key(home.secrets_path, resolved_api_key)

    # Validate the endpoint/model before declaring success so a bad model or
    # key is caught during setup, not later during indexing.
    config.embedding.api_key = resolved_api_key
    try:
        validate_embedding_endpoint(
            api_base=config.embedding.api_base,
            model=config.embedding.model,
            api_key=config.embedding.api_key,
        )
    except ConfigError:
        # Still keep the config/secrets on disk; the user can fix and re-check.
        print(
            f"Setup wrote configuration to {home.config_path} and the key to "
            f"{home.secrets_path}, but the endpoint validation failed. Run "
            "'caracal-rag check' to retry once the service is reachable.",
            file=sys.stderr,
        )
        raise
    print(f"Configured {home.config_path} (model={config.embedding.model!r})")
    print(f"Stored FreeLLMAPI API key at {home.secrets_path} (mode 0600)")
    print("Run 'caracal-rag check' to validate connectivity.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        run_index(args.source, args.config)
    elif args.command == "check":
        run_check(args.config)
    elif args.command == "mcp":
        run_mcp(args.host, args.port, args.config)
    elif args.command == "setup":
        run_setup(args.model, args.api_base)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
