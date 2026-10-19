# caracal-rag

Caracal RAG - a retrieval-augmented generation toolkit.

## Project structure

```
src/caracal_rag/          package source (src layout)
tests/                    pytest test suite
install.sh                WSL/Linux installer into ~/.caracal
.github/workflows/ci.yml  CI: lint (ruff) + tests (pytest) on PRs and main
```

## What is caracal-rag?

`caracal-rag` provides a local retrieval-augmented generation pipeline that downloads knowledge sources, chunks documents, generates embeddings through FreeLLMAPI (or any OpenAI-compatible proxy), and stores vectors in a remote Chroma database. It also exposes an MCP server so an LLM agent can search indexed knowledge.

## Quick Start (Installation)

To install Caracal RAG into `~/.caracal` on WSL/Linux:

```bash
./install.sh

# Make the installed CLI available in this shell; add this line to ~/.bashrc if desired.
export PATH="$HOME/.caracal/bin:$PATH"
```

The installer creates an isolated Python environment at `~/.caracal/venv`, installs the package non-editably into it, places a wrapper at `~/.caracal/bin/caracal-rag`, and creates `~/.caracal/config` and `~/.caracal/index`. It copies `config/sources.example.yaml` to `~/.caracal/config/sources.yaml` only on first install and preserves existing configuration, secrets and index metadata on reruns. Re-running it from an updated source checkout upgrades the package in the virtual environment without switching to a system-wide Python install.

Configure the model and API key, then validate connectivity:

```bash
# The API key is prompted with input hidden; it is not placed in shell history.
caracal-rag setup --model bge-m3

# Validate connectivity to FreeLLMAPI and Chroma.
caracal-rag check
```

For a non-interactive shell, read the key without echoing it and let the CLI persist it privately:

```bash
read -rsp "FreeLLMAPI API key: " CARACAL_EMBEDDING_API_KEY; echo
export CARACAL_EMBEDDING_API_KEY
caracal-rag setup --model bge-m3
unset CARACAL_EMBEDDING_API_KEY
```

## Commands

- `caracal-rag setup [--model <family>] [--api-base <url>]`: configure the model, endpoint and secrets under `~/.caracal`; the API key and the access token are requested via hidden prompts if they are not already configured. Avoid passing keys or tokens as command-line arguments.
- `caracal-rag check`: validate active configuration & connectivity to FreeLLMAPI and Chroma. Also validates the access-token policy, so a deployment whose protected interface would start broken is caught before the MCP server is started.
- `caracal-rag index [--source <name>]`: build/refresh the active index under `~/.caracal`.
- `caracal-rag mcp`: start the stdio MCP server for agent integration.

## Configuration & Secrets

The active configuration lives under `~/.caracal`:
- `~/.caracal/config/sources.yaml` — embedding model, base URL, Chroma host/port, knowledge sources, and the `access_token` policy. (No secrets).
- `~/.caracal/config/secrets.yaml` — credentials (`embedding.api_key` and `access_token.token`), protected with file mode `0600`.
- `~/.caracal/index/manifest.json` — durable record of the collection and embedding space used to build the index.

Secrets can also be supplied via environment variables, which take precedence over `secrets.yaml`:
- `CARACAL_EMBEDDING_API_KEY` — the FreeLLMAPI API key.
- `CARACAL_ACCESS_TOKEN` — the access token that protects the MCP interface.
- `CARACAL_MCP_TRANSPORT` — `stdio` (default) or `http` transport for the MCP server.
- `CARACAL_MCP_HOST`, `CARACAL_MCP_PORT`, `CARACAL_MCP_PATH` — bind address for the `http` transport.

### Access Token Protection

The MCP interface is protected by an access token. The raw token is **never** a tool argument, tool schema field, or tool result: it is supplied out-of-band through the transport, so it cannot leak into a model-visible surface, LLM transcript, or shell history.

- **stdio transport** (the default): a local, unauthenticated interface. It is intended for a trusted host where the agent process itself is the only caller. No auth channel exists on stdio, so the token is not presented at all.
- **http transport**: the interface is exposed behind the MCP SDK's `BearerAuthMiddleware`, which demands an `Authorization: Bearer <token>` header on every request. Missing or invalid tokens are rejected with `401`; valid tokens authenticate the session.

The policy is configured in `sources.yaml`:

```yaml
mcp:
  transport: http        # or 'stdio'
  host: 127.0.0.1        # http bind host
  port: 8000             # http bind port
  path: /mcp             # http mount path
access_token:
  enabled: true   # set to false only to explicitly disable protection
```

The token value itself is never written into `sources.yaml`; it lives in `secrets.yaml` under `access_token.token` (or in the `CARACAL_ACCESS_TOKEN` environment variable). It must be at least 8 characters long. Comparisons are constant-time via `secrets.compare_digest`, and errors never echo the configured token.

### FreeLLMAPI Embedding Models

FreeLLMAPI exposes embedding models via its OpenAI-compatible `/v1/embeddings` endpoint. See the [FreeLLMAPI embeddings API documentation](https://github.com/tashfeenahmed/freellmapi/blob/main/docs/en/api/01-rest-api.md#embeddings) and check **Models → Embeddings** in your own deployment for available families. Example families include:
- `bge-m3` (1024-dim, default recommended)
- `qwen3-embedding-0.6b` (1024-dim)
- `nv-embedqa-e5-v5` (1024-dim)
- `embed-v4.0` (1536-dim)
- `llama-nemotron-embed-1b-v2` (2048-dim)
- `gemini-embedding-001` (3072-dim)
- `embeddinggemma-300m` (768-dim)

`embedding.model` is configured explicitly; `bge-m3` is the shipped template value and can be changed to a family enabled in your deployment. Do not use `auto`: this client must know the effective family so it can protect the existing index if FreeLLMAPI defaults change. Vectors from different families are not interchangeable, even if dimensions happen to match.

### Source Types

Three source types are supported in `sources.yaml`:

* `markdown` – index individual markdown files by raw URLs.
* `github_md_doc_dir` – recursively fetch `.md` files from a GitHub repository directory (e.g., Fiteo API docs at `mariuszmarzec/fiteo`, `docs/api`, `master`).
* `local` – index all `.md` files recursively from a local folder.

## Running the Embedding Pipeline

Build or update the index:

```bash
caracal-rag index
```

To index a single source:

```bash
caracal-rag index --source fiteo-docs
```

## Running the RAG Server (MCP)

Start the persistent stdio MCP server (local, unauthenticated interface):

```bash
caracal-rag mcp
```

Or expose the interface behind an out-of-band Bearer access-token channel:

```bash
caracal-rag mcp --transport http --host 127.0.0.1 --port 8000 --path /mcp
# or set CARACAL_MCP_TRANSPORT=http and the bind env vars
```

The MCP server exposes the `search_api(query, top_k=5)` tool to connected agents (OpenCode, OpenClaw, etc.). The raw access token is never a tool argument: on the `http` transport every request must present a valid `Authorization: Bearer <token>` header, and missing or invalid tokens are rejected with `401`. On the `stdio` transport the interface is local and unauthenticated by design.

## Indexing Consistency & Manifest Validation

- **Manifest Pre-flight**: Before indexing or searching, Caracal checks `~/.caracal/index/manifest.json` against the active config. If the configured model differs from the model that built the index, the operation is refused with rebuild guidance to prevent mixing embedding spaces.
- **Unchanged documents** are skipped.
- **Changed documents** are refreshed atomically via staging rows and promoted upon success.
- **Deleted documents** are removed from Chroma.
- **Transient fetch failures** preserve existing vectors without deleting data.


## Updating, rotating credentials, rebuilding and uninstalling

### Upgrade

From the Caracal source checkout, pull the version you want to install and rerun the installer:

```bash
git pull
./install.sh
```

The installer updates the package in `~/.caracal/venv` from the checkout, but preserves `~/.caracal/config/sources.yaml`, `secrets.yaml`, and the local index manifest. The installed runtime uses the virtual environment and active configuration under `~/.caracal`; it does not import Python modules from the checkout.

### Rotate credentials

Run `caracal-rag setup` and enter the new key and token at the hidden prompts. In a scripted shell, use non-echoing prompts rather than putting secrets in literal commands:

```bash
read -rsp "New FreeLLMAPI API key: " CARACAL_EMBEDDING_API_KEY; echo
read -rsp "New access token: " CARACAL_ACCESS_TOKEN; echo
export CARACAL_EMBEDDING_API_KEY CARACAL_ACCESS_TOKEN
caracal-rag setup --model bge-m3
unset CARACAL_EMBEDDING_API_KEY CARACAL_ACCESS_TOKEN
```

The CLI stores the key and the token in `~/.caracal/config/secrets.yaml` with mode `0600`. Alternatively, `CARACAL_EMBEDDING_API_KEY` and `CARACAL_ACCESS_TOKEN` can be supplied to the runtime environment; do not place secrets in scripts committed to Git or in command-line arguments. Rotating a credential is safe: existing configuration and the other secret are preserved.

### Change the embedding model safely

Do not point a populated collection at a different embedding family. The manifest and per-chunk metadata checks intentionally refuse an unknown, legacy, mixed or mismatched embedding space. To rebuild with a different explicit family, choose a new Chroma collection name in `~/.caracal/config/sources.yaml`, set `embedding.model` to that family's exact name, run `caracal-rag check`, then run `caracal-rag index`. Verify search/MCP against the new collection before retiring the old collection. This keeps the existing index available for rollback.

The embedding **dimension** is checked the same way: `caracal-rag check` probes the configured model and refuses a manifest or live collection whose stored vectors differ in dimensionality, even when the family name agrees. Two models that happen to agree on dimension are still different embedding spaces and must never be mixed.

### Uninstall

Back up configuration, secrets and the local manifest before removing the installation. This removes the local package, configuration, credentials and local state; it does not delete the remote Chroma collection.

```bash
cp -a "$HOME/.caracal" "$HOME/caracal-backup"
rm -rf "$HOME/.caracal"
```
