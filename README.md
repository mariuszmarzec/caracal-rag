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

`caracal-rag` provides a local retrieval-augmented generation pipeline that downloads knowledge sources, chunks documents, generates embeddings through an OpenAI-compatible endpoint, and stores vectors in Chroma. The recommended local setup serves `qllama/multilingual-e5-base` from Ollama through a `multilingual-e5-base` alias in LiteLLM.

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
caracal-rag setup --model multilingual-e5-base --api-base http://127.0.0.1:4000/v1

# Validate connectivity to LiteLLM and Chroma.
caracal-rag check
```

For a non-interactive shell, read the key without echoing it and let the CLI persist it privately:

```bash
read -rsp "LiteLLM API key: " CARACAL_EMBEDDING_API_KEY; echo
export CARACAL_EMBEDDING_API_KEY
caracal-rag setup --model multilingual-e5-base --api-base http://127.0.0.1:4000/v1
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
- `CARACAL_EMBEDDING_API_KEY` — API key for the configured OpenAI-compatible embedding endpoint (the LiteLLM master key for the local Ollama setup).
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
  port: 3500             # http bind port
  path: /mcp             # http mount path
access_token:
  enabled: true   # set to false only to explicitly disable protection
```

The token value itself is never written into `sources.yaml`; it lives in `secrets.yaml` under `access_token.token` (or in the `CARACAL_ACCESS_TOKEN` environment variable). It must be at least 8 characters long. Comparisons are constant-time via `secrets.compare_digest`, and errors never echo the configured token.

### Local multilingual-e5-base through LiteLLM

The default local setup uses Ollama model `qllama/multilingual-e5-base` and exposes it through LiteLLM under the stable alias `multilingual-e5-base`. Add this entry under `model_list` in `~/litellm/config.yaml`:

```yaml
  - model_name: multilingual-e5-base
    litellm_params:
      model: ollama/qllama/multilingual-e5-base
      api_base: http://localhost:11434
      api_key: ollama-local
```

Pull the model once on each machine:

```bash
ollama pull qllama/multilingual-e5-base
```

Configure Caracal to call the LiteLLM OpenAI-compatible embeddings endpoint at `http://127.0.0.1:4000/v1`, with `embedding.model: multilingual-e5-base`. Store the LiteLLM proxy key in `CARACAL_EMBEDDING_API_KEY` or `~/.caracal/config/secrets.yaml`; the key is not the Ollama key used by the upstream model entry.

`multilingual-e5-base` produces 768-dimensional vectors and expects task-specific prefixes for retrieval. Caracal automatically sends `passage: ` before document chunks and `query: ` before search queries. Do not add those prefixes manually to source files or queries.

FreeLLMAPI and other OpenAI-compatible embedding providers remain supported when their endpoint and model alias are configured explicitly. Never use `auto`: the manifest records the selected model, and vectors from different embedding families are not interchangeable even when their dimensions happen to match.

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
caracal-rag mcp --transport http --host 127.0.0.1 --port 3500 --path /mcp
# or set CARACAL_MCP_TRANSPORT=http and the bind env vars
```

The MCP server exposes the `search_api(query, top_k=5)` tool to connected agents (OpenCode, OpenClaw, etc.). The raw access token is never a tool argument: on the `http` transport every request must present a valid `Authorization: Bearer <token>` header, and missing or invalid tokens are rejected with `401`. On the `stdio` transport the interface is local and unauthenticated by design.

### MCP lifecycle after updates

`update.sh` validates and activates a new Python venv generation, but it does **not** restart an MCP process or its host. The actual owner of an MCP process depends on how it was launched; Caracal does not maintain a registry of process IDs or a dedicated service manager, so it cannot safely infer which process should be restarted.

- **`stdio` transport:** the MCP server process is normally spawned by the MCP client/host. After updating, use that host's own UI/configuration to reload or restart the Caracal MCP connection or the relevant host session. Restart the process that owns the session; do not kill a process by fuzzy name.
- **`http` transport (default):** Caracal's HTTP MCP server is a separate server process. Restart it using the process manager or terminal that started that specific Caracal server. If it is managed by a dedicated service in your deployment, use that service's documented restart command and verify its status before reconnecting. The update script does not assume that such a service exists.
- If the service's launcher or ownership is unclear, identify it first instead of restarting OpenClaw, OpenCode, Manul, or other unrelated processes. The post-update message tells you which transport-specific action applies; no restart is claimed to have happened automatically.

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
3. Creates a separate candidate venv under `~/.caracal/venvs/` and runs `install.sh` with the matching checked-in Python dependency constraints lock. The active `~/.caracal/venv` and its stable CLI wrapper are not changed while package installation is in progress.
4. Verifies the candidate Python and `caracal-rag` entry point exist, then runs `caracal-rag check` directly from the candidate before activating it.
5. Promotes the validated candidate by atomically switching `~/.caracal/venv` to a symlink to the new generation. The previous environment is retained for rollback; older generations are not automatically deleted.

If you have local edits you intentionally want to install without pulling from Git, run:

```bash
bash ./update.sh --no-pull
```

To skip the connectivity check (for example, when the embedding service or Chroma is temporarily unreachable):

```bash
bash ./update.sh --skip-check
```

You can combine the options: `bash ./update.sh --no-pull --skip-check`. The default update refuses to pull if the checkout has modified, staged or untracked files. Commit/stash those files first, or use `--no-pull` to install the current checkout as-is. Even with `--skip-check`, the updater requires a runnable candidate Python and `caracal-rag` entry point before promotion.

#### Candidate validation and rollback

The candidate lives at a unique path under `~/.caracal/venvs/`, not at the active venv path. The installer applies the checked-in `constraints-py310.lock` through `constraints-py314.lock` file matching the candidate Python interpreter, while `CARACAL_HOME` still points at the existing local configuration. The update uses the candidate entry point to run the post-install check before switching the active environment.

If dependency installation, package installation, entry-point validation, or `caracal-rag check` fails, the active venv is not promoted or overwritten and the failed candidate is removed. The update reports the original error and exits nonzero. If activation itself fails, the updater attempts to restore the previous active target; if that rollback also fails it reports both the failure and the paths that must be inspected, and it preserves any candidate that may still be active rather than deleting a possibly working environment.

For a legacy installation where `~/.caracal/venv` is a real directory, the first successful transactional update retains that directory under `~/.caracal/venvs/caracal-venv-legacy-...` and turns `~/.caracal/venv` into a symlink. This keeps old console-script shebangs resolving correctly if rollback is needed. Each later successful update retains the previous generation too; the updater deliberately does not prune old generations automatically. After a new generation has been verified over time, inspect the active target with `readlink -f "$HOME/.caracal/venv"` before manually removing only obsolete generations. Never remove the active target or the fallback you still need.

The script and installer preserve `~/.caracal/config/sources.yaml`, `~/.caracal/config/secrets.yaml`, and the local index manifest; they do not rebuild the index, delete remote Chroma data, or roll back user configuration. No MCP process is restarted by the update. After a successful update, follow [MCP lifecycle after updates](#mcp-lifecycle-after-updates) for the appropriate manual action for the configured transport.

The installed runtime uses the active venv and configuration under `~/.caracal`; it does not import Python modules directly from the checkout. You can also run `bash ./install.sh` manually to reinstall the current checkout without pulling or running the post-install connectivity check. A manual install targets the currently active venv; use `update.sh` when you need candidate validation and rollback protection.

### Rotate credentials

Run `caracal-rag setup` and enter the new key and token at the hidden prompts. In a scripted shell, use non-echoing prompts rather than putting secrets in literal commands:

```bash
read -rsp "New embedding endpoint API key: " CARACAL_EMBEDDING_API_KEY; echo
read -rsp "New access token: " CARACAL_ACCESS_TOKEN; echo
export CARACAL_EMBEDDING_API_KEY CARACAL_ACCESS_TOKEN
caracal-rag setup --model multilingual-e5-base --api-base http://127.0.0.1:4000/v1
unset CARACAL_EMBEDDING_API_KEY CARACAL_ACCESS_TOKEN
```

The CLI stores the key and the token in `~/.caracal/config/secrets.yaml` with mode `0600`. Alternatively, `CARACAL_EMBEDDING_API_KEY` and `CARACAL_ACCESS_TOKEN` can be supplied to the runtime environment; do not place secrets in scripts committed to Git or in command-line arguments. Rotating a credential is safe: existing configuration and the other secret are preserved.

### Change the embedding model safely

Do not point a populated collection at a different embedding family. The manifest and per-chunk metadata checks intentionally refuse an unknown, legacy, mixed or mismatched embedding space. To rebuild with a different explicit family, choose a new Chroma collection name in `~/.caracal/config/sources.yaml`, set `embedding.model` to that family's exact name, run `caracal-rag check`, then run `caracal-rag index`. Verify search/MCP against the new collection before retiring the old collection. This keeps the existing index available for rollback.

The embedding **dimension** is checked the same way: `caracal-rag check` probes the configured model and refuses a manifest or live collection whose stored vectors differ in dimensionality, even when the family name agrees. Two models that happen to agree on dimension are still different embedding spaces and must never be mixed.

## Backup and restore

Caracal state lives locally under `~/.caracal`, but the vectors are stored by the separate Chroma service configured in `sources.yaml` (`chroma.host`, `chroma.port`, `chroma.collection`). A local-state backup is **not** a complete RAG backup unless the matching Chroma data is backed up separately. `backup.sh` deliberately never connects to Chroma or copies/deletes vector data.

### Back up local state

Run from a Caracal checkout:

```bash
bash ./backup.sh
# or choose an explicit destination outside the checkout:
bash ./backup.sh --output "$HOME/caracal-backups/pre-upgrade-2026-10-21"
```

The default is a new timestamped folder under `~/caracal-backups`; `CARACAL_BACKUP_ROOT` can select a different parent. The output directory and its subdirectories are mode `0700`; saved files are mode `0600`. The script fails instead of overwriting an existing destination, refuses symlinked state directories/files, and never prints secret values. The backup contains:

- `config/sources.yaml` — source, embedding, Chroma collection and MCP configuration;
- `config/secrets.yaml` — the LiteLLM API key and MCP access token, if configured;
- `index/manifest.json` — included only if it exists, identifying the collection and embedding space recorded by Caracal;
- `BACKUP-INFO.txt` — format/time and an explicit `chroma_vectors=included:NO` marker.

Do not commit or upload these files to a public repository; the backup includes credentials. Store an additional copy on encrypted storage with access controls appropriate for secrets.

### Back up Chroma separately

Caracal currently connects through Chroma's `HttpClient`; it does not know whether the server uses a local persistent path, a Docker bind mount/named volume, a remote VM, a distributed deployment, or a managed Chroma service. For that reason there is no safe universal Chroma-copy command in `backup.sh`.

- **Self-hosted single-node / Docker:** determine the actual persistent data path from the Chroma launch command/configuration. The official Docker example mounts a host path to `/data`; deployments may instead use a named volume or another path. Schedule a maintenance window, stop the Chroma server cleanly, take a filesystem/volume snapshot or archive of the *actual persisted data*, then restart it. Do not copy a live SQLite/HNSW data directory and assume the result is consistent. Restore that snapshot to the expected storage path while Chroma is stopped, then start the same compatible deployment version and verify the collection before reconnecting Caracal. See [Chroma's Docker deployment guide](https://docs.trychroma.com/deployment/docker) and [client/server mode](https://docs.trychroma.com/production/chroma-server/client-server-mode).
- **Managed/Cloud or distributed Chroma:** use the backup/export and recovery procedure provided for that specific service and plan, including all catalog, vector/index and object-storage components it requires. The Caracal scripts cannot make or verify a complete managed-service snapshot.
- **No usable Chroma backup:** the index can be rebuilt from configured sources with `caracal-rag index` after connectivity and model configuration are verified. Re-indexing is not a restore of the old index: source revisions, embeddings, metadata or unavailable documents may differ. Do not run it as an initial troubleshooting step against a collection you are trying to preserve.

Write down the Chroma deployment type, exact persisted path/volume or provider recovery identifier, version, `chroma.collection` name and restore owner alongside the backup. Test the provider/storage restore procedure separately; this repository's tests intentionally never touch a live Chroma database.

### Restore local state

First recover the **matching Chroma collection** using its deployment-specific procedure when you need the old vectors. Then restore the local files into the intended Caracal home:

```bash
bash ./restore.sh "$HOME/caracal-backups/caracal-YYYYMMDDTHHMMSSZ"
```

Restoring into an empty target does not require `--force`. If any target files already exist, the script refuses to change them. Use `--force` only after reviewing the backup: it first saves the existing local config/secrets/manifest to a separate private `pre-restore-*` folder under `~/caracal-backups`, then replaces the requested files. Secrets are restored with mode `0600`. `restore.sh` never starts/stops Chroma, never deletes a collection, and never re-indexes.

Before serving traffic, compare the restored `chroma.collection`, `embedding.model` and `embedding.api_base` settings in `config/sources.yaml` with the `collection`, `embedding_model` and `embedding_api_base` fields in `index/manifest.json`. The collection must be the one recovered from the Chroma snapshot. Do not edit the manifest just to make a mismatch disappear; a mismatch means the local state and vectors may not describe the same embedding space. If the backup predates indexing and has no manifest, follow the first-index setup/rebuild procedure rather than pretending an old collection was restored.

### Verify recovery

1. Put the restored `~/.caracal/bin` on `PATH` and run `caracal-rag check`. Resolve any configuration, credential, embedding endpoint or Chroma connectivity failures before continuing.
2. Start the normal MCP host/process for this installation (OpenCode/OpenClaw for stdio, or the separately managed Caracal HTTP MCP service for HTTP). Restart only the process that actually owns the Caracal MCP session; do not kill processes by name.
3. From a connected MCP client, call `search_api` with a query known to match a document that should exist in the recovered collection. Confirm it returns the expected source/document and useful content. A successful `check` alone proves connectivity, not that the expected vectors were restored. If it returns no expected result, stop and investigate the Chroma snapshot/collection identity before running `caracal-rag index`.
4. If the embedding key or access token changed, rotate it with `caracal-rag setup` (hidden prompts) or the documented environment variables, then repeat the check and authenticated MCP search.

### Uninstall

Back up local state before removing the installation. The commands below remove local package/configuration/credentials; they do not remove remote Chroma data. To create the local backup, use `bash ./backup.sh` above rather than copying secret files with the shell's default permissions.

```bash
rm -rf "$HOME/.caracal"
```
