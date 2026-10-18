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
```

The installer:
1. Creates `~/.caracal/config` and `~/.caracal/index`.
2. Seeds `~/.caracal/config/sources.yaml` from `config/sources.example.yaml` (defaulting to the stable FreeLLMAPI embedding family `bge-m3`).
3. Creates `~/.caracal/config/secrets.yaml` with mode `0600` for your FreeLLMAPI key.

Next, set up your API key and validate connectivity:

```bash
# Provide key interactively or via flag
caracal-rag setup --model bge-m3 --api-key <your-freellmapi-key>

# Validate connectivity to FreeLLMAPI and Chroma
caracal-rag check
```

## Commands

- `caracal-rag setup [--model <family>] [--api-key <key>]`: configure embedding model & secrets under `~/.caracal`.
- `caracal-rag check`: validate active configuration & connectivity to FreeLLMAPI and Chroma.
- `caracal-rag index [--source <name>]`: build/refresh the active index under `~/.caracal`.
- `caracal-rag mcp`: start the stdio MCP server for agent integration.

## Configuration & Secrets

The active configuration lives under `~/.caracal`:
- `~/.caracal/config/sources.yaml` — embedding model, base URL, Chroma host/port, and knowledge sources. (No secrets).
- `~/.caracal/config/secrets.yaml` — credentials (`embedding.api_key`), protected with file mode `0600`.
- `~/.caracal/index/manifest.json` — durable record of the collection and embedding space used to build the index.

Secrets can also be supplied via the environment variable `CARACAL_EMBEDDING_API_KEY`, which takes precedence over `secrets.yaml`.

### FreeLLMAPI Embedding Models

FreeLLMAPI exposes embedding models via its OpenAI-compatible `/v1/embeddings` endpoint. Supported families include:
- `bge-m3` (1024-dim, default recommended)
- `qwen3-embedding-0.6b` (1024-dim)
- `nv-embedqa-e5-v5` (1024-dim)
- `embed-v4.0` (1536-dim)
- `llama-nemotron-embed-1b-v2` (2048-dim)
- `gemini-embedding-001` (3072-dim)
- `embeddinggemma-300m` (768-dim)

`embedding.model` must be set explicitly. There is no implicit default model, ensuring vector spaces are never accidentally mixed.

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

Start the persistent stdio MCP server:

```bash
caracal-rag mcp
```

The MCP server exposes the `search_api(query, top_k=5)` tool to connected agents (OpenCode, OpenClaw, etc.).

## Indexing Consistency & Manifest Validation

- **Manifest Pre-flight**: Before indexing or searching, Caracal checks `~/.caracal/index/manifest.json` against the active config. If the configured model differs from the model that built the index, the operation is refused with rebuild guidance to prevent mixing embedding spaces.
- **Unchanged documents** are skipped.
- **Changed documents** are refreshed atomically via staging rows and promoted upon success.
- **Deleted documents** are removed from Chroma.
- **Transient fetch failures** preserve existing vectors without deleting data.
