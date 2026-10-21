# caracal-rag

Caracal RAG - a retrieval-augmented generation toolkit.

## Project structure

```
src/caracal_rag/          package source (src layout)
tests/                    pytest test suite
install.sh                WSL/Linux installer into ~/.caracal
update.sh                 Safe Git update + reinstall + validation
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

## Reproducible dependency installs

The supported WSL/Linux interpreters are Python 3.10 through 3.14. The checked-in
`constraints-py310.lock` through `constraints-py314.lock` files pin the complete
resolved runtime, development/test, and build dependency graph for each Python
minor version. The installer detects the Python version inside the actual
`~/.caracal/venv` and applies its matching constraints file. Fresh installs and
updates therefore use the same approved package versions for the same Python
minor; the installer still installs into the isolated virtual environment and
never installs project dependencies system-wide.

CI runs lock verification and the test suite on all five supported Python
versions. The lock files are generated from `pyproject.toml` using
[pip-tools](https://pip-tools.pypa.io/), with the generator version pinned in
`tools/refresh-dependencies.sh`. To refresh every lock, install Python 3.10,
3.11, 3.12, 3.13 and 3.14 (the corresponding `python3.x` commands must be on
`PATH`) and run:

```bash
# Re-resolve changed/new requirements while retaining compatible existing pins.
bash tools/refresh-dependencies.sh

# Update one package and any pins needed to keep its graph compatible.
bash tools/refresh-dependencies.sh --upgrade-package chromadb

# Intentionally refresh all compatible pins; inspect the diff and run CI.
bash tools/refresh-dependencies.sh --upgrade

# Verify all checked-in locks match pyproject.toml without changing them.
bash tools/refresh-dependencies.sh --check
```

The refresh/check script creates temporary virtual environments for the pinned
lock-generation tool and removes them when it exits. `--check` uses each
existing lock as a constraint and compares the resolved pin set, so adding,
removing, or changing declared dependencies is detected by CI. A targeted run
such as `bash tools/refresh-dependencies.sh --check --python 3.12` verifies
just one interpreter's lock.

For security fixes, prefer reviewing advisories and updating the affected
package deliberately (for example with `--upgrade-package package-name`) rather
than blindly downgrading a pin or refreshing the entire graph. Review the lock
diff and run the full matrix before merging an intentional dependency update.
Python versions outside the checked-in lock matrix are rejected by the
installer until a matching lock is generated and verified.

## Commands

- `caracal-rag setup [--model <family>] [--api-base <url>]`: configure the model, endpoint and secrets under `~/.caracal`; the API key and the access token are requested via hidden prompts if they are not already configured. Avoid passing keys or tokens as command-line arguments.
- `caracal-rag check`: validate active configuration & connectivity to FreeLLMAPI and Chroma. Also validates the access-token policy, so a deployment whose protected interface would start broken is caught before the MCP server is started.
- `caracal-rag index [--source <name>]`: build/refresh the active index under `~/.caracal`.
- `caracal-rag mcp`: start the MCP server for agent integration. The default transport is authenticated HTTP; stdio is an explicit opt-in local/unauthenticated interface.

### MCP search input and error handling

The `search_api` tool requires `query` to be a non-empty string after trimming whitespace. `top_k` must be a real integer from 1 through 20 inclusive; the default is 5. Booleans, fractional values, numeric strings, null, zero, negatives, and values above 20 are rejected rather than coerced. Invalid arguments return a structured tool error with code `invalid_query` or `invalid_top_k`.

If embedding or Chroma operations fail, the tool returns a generic `search_failed` error with a pointer to `caracal-rag check`. Raw backend exception text is intentionally omitted from tool results because it may include private endpoint details or credentials. After checking connectivity and configuration, retry the request; handled errors should not terminate the MCP server session.

CI runs the real authenticated HTTP transport against the pinned MCP SDK version in the dependency lock (MCP 2.3.0). Deterministic handler tests also exercise the decorator-style API used by the MCP SDK 1.x family, so both handler registration paths share the same validation behavior.

## Configuration & Secrets

The active configuration lives under `~/.caracal`:
- `~/.caracal/config/sources.yaml` — embedding model, base URL, Chroma host/port, knowledge sources, and the `access_token` policy. (No secrets).
- `~/.caracal/config/secrets.yaml` — credentials (`embedding.api_key` and `access_token.token`), protected with file mode `0600`.
- `~/.caracal/index/manifest.json` — durable record of the collection and embedding space used to build the index.

Secrets can also be supplied via environment variables, which take precedence over `secrets.yaml`:
- `CARACAL_EMBEDDING_API_KEY` — the FreeLLMAPI API key.
- `CARACAL_ACCESS_TOKEN` — the access token that protects the MCP interface.
- `CARACAL_MCP_TRANSPORT` — `http` (default, authenticated) or `stdio` transport for the MCP server.
- `CARACAL_MCP_HOST`, `CARACAL_MCP_PORT`, `CARACAL_MCP_PATH` — bind address for the `http` transport.

### Access Token Protection

The MCP interface is protected by an access token. The raw token is **never** a tool argument, tool schema field, or tool result: it is supplied out-of-band through the transport, so it cannot leak into a model-visible surface, LLM transcript, or shell history.

- **http transport** (the default): the interface is exposed behind the MCP SDK's `BearerAuthMiddleware`, which demands an `Authorization: Bearer <token>` header on every request. Missing or invalid tokens are rejected with `401`; valid tokens authenticate the session.
- **stdio transport** (explicit opt-in): a local, unauthenticated interface. It is intended for a trusted host where the agent process itself is the only caller. No auth channel exists on stdio, so the token is not presented at all. Do not run stdio where the host is shared or untrusted.

The policy is configured in `sources.yaml`:

```yaml
mcp:
  transport: http        # default authenticated; 'stdio' is an explicit opt-in local/unauthenticated interface
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

Start the MCP server. The default is the authenticated HTTP transport, which demands a valid `Authorization: Bearer <token>` header on every request:

```bash
caracal-rag mcp
```

Or run the local, unauthenticated stdio interface as an explicit opt-in for a trusted host:

```bash
caracal-rag mcp --transport stdio
# or set CARACAL_MCP_TRANSPORT=stdio
```

To change the bind address of the HTTP transport:

```bash
caracal-rag mcp --transport http --host 127.0.0.1 --port 8000 --path /mcp
# or set CARACAL_MCP_TRANSPORT=http and the bind env vars
```

The MCP server exposes the `search_api(query, top_k=5)` tool to connected agents (OpenCode, OpenClaw, etc.). The raw access token is never a tool argument: on the `http` transport every request must present a valid `Authorization: Bearer <token>` header, and missing or invalid tokens are rejected with `401`. On the `stdio` transport the interface is local and unauthenticated by design.

## Indexing Consistency & Manifest Validation

- **Manifest Pre-flight**: Before indexing or searching, Caracal checks `~/.caracal/index/manifest.json` against the active config. If the configured model differs from the model that built the index, the operation is refused with rebuild guidance to prevent mixing embedding spaces.
- **Unchanged documents** are skipped.
- **Changed documents** use per-document staging rows and rollback. A replacement is embedded before the previous committed chunks are changed; temporary rows are marked and excluded from search.
- **Deleted documents** are reconciled only after a complete source inventory has loaded and every discovered document has indexed successfully. If inventory, embedding, or a Chroma write fails, destructive stale-document reconciliation is skipped. The run as a whole is intentionally not transactional: earlier per-document replacements can already have committed before a later document fails. Rerun indexing after resolving the error to finish reconciliation.
- A Chroma stale-row delete can be partially applied if the remote fails mid-request. Such rows were identified as stale from a complete successful inventory; current rows are preserved, the run reports failure, the manifest is not updated, and a retry cleans up remaining stale rows.
- **Transient fetch/listing failures** preserve old rows for documents that were not fully inventoried. Successfully loaded documents may still be refreshed, but the index command reports an incomplete run and does not update the manifest.
- **Unexpectedly empty sources**: when a source previously has committed chunks but now returns zero documents, indexing aborts before indexing or deleting anything. Check the source path, URL/repository, and branch first. Only for an intentionally empty source whose old rows should be removed, set `allow_empty: true` on that source in `sources.yaml`. For example:

  ```yaml
  sources:
    - name: archived-docs
      type: local
      path: ./archived-docs
      allow_empty: true
  ```

  The default is `allow_empty: false`; the flag is an explicit opt-in to clearing that source when its inventory is empty.


## Updating, rotating credentials, rebuilding and uninstalling

### Upgrade / reinstall

Run the update script from the root of your Caracal source checkout:

```bash
bash ./update.sh
```

By default, the script:
1. Requires a clean Git worktree and a current branch with an upstream configured.
2. Runs `git pull --ff-only` so it will not create a merge commit or silently resolve divergent history.
3. Runs `install.sh` to reinstall the package and its Python dependencies into `~/.caracal/venv`.
4. Runs `caracal-rag check` using the installed CLI to validate configuration and connectivity.

If you have local edits you intentionally want to install without pulling from Git, run:

```bash
bash ./update.sh --no-pull
```

To skip the connectivity check (for example, when FreeLLMAPI or Chroma is temporarily unreachable):

```bash
bash ./update.sh --skip-check
```

You can combine the options: `bash ./update.sh --no-pull --skip-check`. The default update refuses to pull if the checkout has modified, staged or untracked files. Commit/stash those files first, or use `--no-pull` to reinstall the current checkout as-is.

The script and installer preserve `~/.caracal/config/sources.yaml`, `~/.caracal/config/secrets.yaml`, and the local index manifest; they do not rebuild the index or delete the remote Chroma collection. After a successful update, restart the process or host that launches the Caracal MCP server (such as OpenClaw/OpenCode) so it imports the new Python code. The script cannot safely restart an MCP host whose lifecycle is managed separately.

The installed runtime uses the virtual environment and active configuration under `~/.caracal`; it does not import Python modules directly from the checkout. You can also run `bash ./install.sh` manually to reinstall the current checkout without pulling or running the post-install check.

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
