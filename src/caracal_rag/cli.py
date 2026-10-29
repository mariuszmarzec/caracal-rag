from __future__ import annotations

import argparse
import asyncio
import os
import sys

from caracal_rag.auth import redact_token
from caracal_rag.benchmark import audit_index, benchmark_retrieval, write_report
from caracal_rag.config import (
    ACCESS_TOKEN_ENV,
    EMBEDDING_API_KEY_ENV,
    AccessTokenConfig,
    AppConfig,
    ChromaConfig,
    ConfigError,
    EmbeddingConfig,
    load_active_config,
    resolve_home,
    write_config,
    write_secret_access_token,
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

    audit_parser = subparsers.add_parser(
        "audit", help="Inspect the existing index without calling the embedding API"
    )
    audit_parser.add_argument(
        "--config",
        default=None,
        help="Path to the active configuration file (default: the installed configuration)",
    )
    audit_parser.add_argument(
        "--output", default=None, help="Write the detailed JSON report to this local file"
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Measure retrieval quality using a curated local JSON dataset"
    )
    benchmark_parser.add_argument(
        "--dataset", required=True, help="Path to a local benchmark dataset JSON file"
    )
    benchmark_parser.add_argument(
        "--top-k",
        type=int,
        choices=range(1, 21),
        default=10,
        help="Number of retrieval results to evaluate (1-20)",
    )
    benchmark_parser.add_argument(
        "--config",
        default=None,
        help="Path to the active configuration file (default: the installed configuration)",
    )
    benchmark_parser.add_argument(
        "--output", default=None, help="Write the detailed JSON report to this local file"
    )

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
        "--transport",
        default=None,
        choices=["stdio", "http"],
        help="MCP transport (default: $CARACAL_MCP_TRANSPORT, then 'http'). "
        "http exposes the interface behind an out-of-band Bearer access-token "
        "channel; stdio is a local, unauthenticated interface for a trusted host.",
    )
    mcp_parser.add_argument(
        "--host",
        default=None,
        help="MCP server host for the http transport (default: $CARACAL_MCP_HOST, "
        "then 127.0.0.1).",
    )
    mcp_parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="MCP server port for the http transport (default: $CARACAL_MCP_PORT, "
        "then 3500).",
    )
    mcp_parser.add_argument(
        "--path",
        default=None,
        help="MCP server path for the http transport (default: $CARACAL_MCP_PATH, "
        "then /mcp).",
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
    # The access token is intentionally NOT a CLI argument: command-line
    # arguments leak through shell history and process listings. It is supplied
    # through the hidden prompt or the CARACAL_ACCESS_TOKEN environment var.
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
    indexing_complete = indexer.run()
    if indexing_complete is False:
        raise RuntimeError(
            "Indexing loaded the available documents but source inventory was incomplete; "
            "stale-row reconciliation was skipped and the manifest was not updated. "
            "Resolve the source loading failure and rerun indexing."
        )
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

    The access-token policy is validated here too, so a deployment whose
    protected interface would start in a broken state is caught before the
    MCP server is started.
    """
    config: AppConfig | None = None
    try:
        config = load_active_config(config_path)
        _ensure_api_key(config)
        _ensure_access_token(config)
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
        print(f"access_token_enabled={config.access_token.enabled}")
        print(f"access_token_configured={bool(config.access_token.token)}")

        client = _chroma_client(
            host=config.chroma.host,
            port=config.chroma.port,
            ssl=config.chroma.ssl,
        )
        heartbeat = client.heartbeat()
        print(f"chroma_heartbeat={heartbeat}")

        print(f"embedding_dim={observed_dim}")
        print(
            f"mcp_transport={config.mcp.transport} "
            f"mcp_host={config.mcp.host} mcp_port={config.mcp.port} "
            f"mcp_path={config.mcp.path}"
        )
        print("ok")
    except Exception as exc:  # noqa: BLE001 - surface any connectivity failure
        secret = (
            config.embedding.api_key
            if config is not None
            else os.getenv(EMBEDDING_API_KEY_ENV)
        )
        token = config.access_token.token if config is not None else os.getenv(ACCESS_TOKEN_ENV)
        print(redact_secret(redact_token(f"check failed: {exc}", token), secret), file=sys.stderr)
        sys.exit(1)


def run_mcp(
    transport_override: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
    path_override: str | None = None,
    config_path: str | None = None,
) -> None:
    from caracal_rag.mcp import CaracalMcpServer

    config = load_active_config(config_path)
    _ensure_api_key(config)
    _ensure_access_token(config)
    # Probe the live dimension before the MCP server starts serving. The
    # manifest and collection are validated against it, so a dimension drift
    # is refused before any query vector is generated.
    observed_dim = _probe_dim(config)
    _assert_index_safe(config, observed_dim=observed_dim)

    # Transport/host/port/path are read from the env-aware config by default;
    # CLI overrides win for connection debugging.
    if transport_override is not None:
        config.mcp.transport = transport_override
    if host_override is not None:
        config.mcp.host = host_override
    if port_override is not None:
        config.mcp.port = port_override
    if path_override is not None:
        config.mcp.path = path_override

    server = CaracalMcpServer(config)

    async def _run() -> None:
        if config.mcp.transport == "http":
            await server.run_http_async()
        else:
            await server.run_stdio_async()

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


def _ensure_access_token(config: AppConfig) -> None:
    """Fail early with actionable guidance when the access token is unusable.

    ``enabled`` defaults to ``True``, so a deployment whose protected interface
    would start broken is caught at ``check``/``mcp`` startup instead of at the
    first request. The token is never echoed: the message only names where to
    obtain or rotate it.
    """
    if not config.access_token.enabled:
        return
    token = config.access_token.token
    if not isinstance(token, str) or len(token) < 8:
        raise ConfigError(
            "Access authentication is enabled but no valid access token is "
            "configured. Provide one by running 'caracal-rag setup', by "
            f"exporting {ACCESS_TOKEN_ENV}, or by setting `access_token.token` "
            "in the secrets file under your Caracal home "
            "(~/.caracal/config/secrets.yaml). The token must be at least 8 "
            "characters long."
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
    access_token: str | None = None,
) -> None:
    """Create or update the active installed configuration and secrets.

    Idempotent: existing configuration and secrets are preserved and only the
    fields the user explicitly provides are updated. The API key and the
    access token are read from hidden prompts (not echoed) when not given on
    the command line, and are never passed as command-line arguments.
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

    # The access token protects every interface path. It is required by
    # default; the only way to change that is an explicit `enabled: false` in
    # the active configuration. The token is prompted once on first setup and
    # preserved on reruns unless the user explicitly supplies a new one.
    resolved_token = (
        access_token
        or (existing.access_token.token if existing else None)
        or os.getenv(ACCESS_TOKEN_ENV)
    )
    if not resolved_token:
        if interactive:
            import getpass

            prompt = "Caracal RAG access token (input hidden, min 8 chars): "
            resolved_token = getpass.getpass(prompt).strip() or None
        if not resolved_token:
            raise ConfigError(
                "No access token supplied. Every protected Caracal RAG "
                "interface requires a valid token. Enter it at the hidden "
                "prompt, or export "
                f"{ACCESS_TOKEN_ENV} and re-run `caracal-rag setup`. Do not "
                "pass the token as a command-line argument."
            )
    if len(resolved_token) < 8:
        raise ConfigError(
            "The access token is too short. It must be at least 8 characters "
            "long; a shorter value would not be a usable credential."
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
            access_token=AccessTokenConfig(token=resolved_token),
            config_path=home.config_path,
        )

    # Persist the active configuration (no secrets) and the secrets file.
    write_config(config, home.config_path)
    write_secret_api_key(home.secrets_path, resolved_api_key)
    write_secret_access_token(home.secrets_path, resolved_token)

    # Validate the endpoint/model before declaring success so a bad model or
    # key is caught during setup, not later during indexing.
    config.embedding.api_key = resolved_api_key
    config.access_token.token = resolved_token
    try:
        validate_embedding_endpoint(
            api_base=config.embedding.api_base,
            model=config.embedding.model,
            api_key=config.embedding.api_key,
        )
    except ConfigError:
        # Still keep the config/secrets on disk; the user can fix and re-check.
        print(
            f"Setup wrote configuration to {home.config_path}, the key to "
            f"{home.secrets_path}, and the access token to "
            f"{home.secrets_path}, but the endpoint validation failed. Run "
            "'caracal-rag check' to retry once the service is reachable.",
            file=sys.stderr,
        )
        raise
    print(f"Configured {home.config_path} (model={config.embedding.model!r})")
    print(f"Stored FreeLLMAPI API key at {home.secrets_path} (mode 0600)")
    print(f"Stored access token at {home.secrets_path} (mode 0600)")
    print("Run 'caracal-rag check' to validate connectivity.")


def _print_audit_summary(report: dict, output_path: str | None) -> None:
    counts = report["counts"]
    print(f"collection={report['collection']}")
    print(f"embedding_model_expected={report['expected_embedding_model']}")
    for name in (
        "total_chunks",
        "committed_chunks",
        "staging_chunks",
        "documents",
        "sources",
        "empty_chunks",
        "missing_embedding_model_metadata",
        "missing_embedding_dimension_metadata",
        "duplicate_chunk_text_groups",
    ):
        print(f"{name}={counts[name]}")
    print(f"manifest_status={report['manifest']['status']}")
    lengths = report["chunk_length_chars"]
    print(
        "chunk_length_chars="
        f"min:{lengths['min']} median:{lengths['median']} mean:{lengths['mean']} "
        f"p95:{lengths['p95']} max:{lengths['max']}"
    )
    for source in report["by_source"]:
        print(
            f"source={source['source']} documents={source['documents']} "
            f"chunks={source['chunks']} characters={source['characters']}"
        )
    for warning in report["warnings"]:
        print(f"warning={warning}")
    if output_path:
        destination = write_report(report, output_path)
        print(f"report={destination}")


def run_audit(config_path: str | None = None, output_path: str | None = None) -> None:
    """Audit stored index content without embedding queries or invoking an LLM."""
    config = load_active_config(config_path)
    report = audit_index(config)
    _print_audit_summary(report, output_path)


def run_benchmark(
    dataset_path: str,
    top_k: int = 10,
    config_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run the local retrieval benchmark; only the configured embedding endpoint is used."""
    config = load_active_config(config_path)
    _ensure_api_key(config)
    report = benchmark_retrieval(config, dataset_path, top_k=top_k)
    print(f"collection={report['collection']}")
    print(f"embedding_model={report['embedding_model']}")
    print(f"cases={report['case_count']} top_k={report['top_k']}")
    print(f"mean_latency_ms={report['mean_latency_ms']}")
    print(f"expected_term_coverage={report['expected_term_coverage']}")
    for name, value in report["metrics"].items():
        print(f"{name}={value}")
    for case in report["cases"]:
        best = case["retrieved"][0] if case["retrieved"] else {}
        best_target = best.get("url") or best.get("document") or "<no results>"
        print(
            f"case={case['id']} hit={case['hit']} recall={case['recall']:.3f} "
            f"mrr={case['reciprocal_rank']:.3f} first_result={best_target}"
        )
    if output_path:
        destination = write_report(report, output_path)
        print(f"report={destination}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "audit":
        run_audit(args.config, args.output)
    elif args.command == "benchmark":
        run_benchmark(args.dataset, args.top_k, args.config, args.output)
    elif args.command == "index":
        run_index(args.source, args.config)
    elif args.command == "check":
        run_check(args.config)
    elif args.command == "mcp":
        run_mcp(
            args.transport,
            args.host,
            args.port,
            args.path,
            args.config,
        )
    elif args.command == "setup":
        run_setup(args.model, args.api_base)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
