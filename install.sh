#!/usr/bin/env bash
# Install Caracal RAG into ~/.caracal and configure stable FreeLLMAPI embeddings.
#
# Idempotent: re-running the script updates the installed configuration and
# secrets in place without touching user-edited fields.
set -euo pipefail

CARACAL_HOME="${CARACAL_HOME:-$HOME/.caracal}"
CARACAL_PYTHON="${CARACAL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_CONFIG="${REPO_ROOT}/config/sources.example.yaml"

CONFIG_DIR="${CARACAL_HOME}/config"
INDEX_DIR="${CARACAL_HOME}/index"
CONFIG_PATH="${CONFIG_DIR}/sources.yaml"
SECRETS_PATH="${CONFIG_DIR}/secrets.yaml"
MANIFEST_PATH="${INDEX_DIR}/manifest.json"

echo "Installing Caracal RAG into ${CARACAL_HOME}"

mkdir -p "${CONFIG_DIR}" "${INDEX_DIR}"

# 1. Install the package into a virtualenv when no project python is set.
if ! "${CARACAL_PYTHON}" -c "import caracal_rag" 2>/dev/null; then
    echo "caracal_rag is not importable from ${CARACAL_PYTHON}; installing with pip."
    "${CARACAL_PYTHON}" -m pip install --quiet -e "${REPO_ROOT}"
fi

# 2. Seed the active configuration from the example shipped with the checkout.
#    Preserve an existing file so user edits are not lost.
if [ ! -f "${CONFIG_PATH}" ]; then
    if [ -f "${EXAMPLE_CONFIG}" ]; then
        cp "${EXAMPLE_CONFIG}" "${CONFIG_PATH}"
        echo "Wrote ${CONFIG_PATH} (from config/sources.example.yaml)"
    else
        cat > "${CONFIG_PATH}" <<'EOF'
embedding:
  model: bge-m3
  api_base: http://localhost:3001/v1

chroma:
  host: localhost
  port: 3400
  collection: caracal-base

sources: []
EOF
        echo "Wrote ${CONFIG_PATH} (built-in template; checkout config not found)"
    fi
else
    echo "Preserving existing ${CONFIG_PATH}"
fi

# 3. Ensure the secrets file exists with restrictive permissions. The API key
#    is never committed; the file is created empty and the user fills it in
#    (or re-runs `caracal-rag setup`).
if [ ! -f "${SECRETS_PATH}" ]; then
    mkdir -p "${CONFIG_DIR}"
    : > "${SECRETS_PATH}"
    chmod 0600 "${SECRETS_PATH}"
    echo "Created ${SECRETS_PATH} (mode 0600; add embedding.api_key or run 'caracal-rag setup')"
else
    chmod 0600 "${SECRETS_PATH}" || true
    echo "Preserving ${SECRETS_PATH} (mode ensured 0600)"
fi

# 4. The manifest is bookkeeping written by indexing; leave it absent until the
#    first index run so consistency checks report "no index yet".
if [ -f "${MANIFEST_PATH}" ]; then
    echo "Preserving existing ${MANIFEST_PATH}"
fi

# 5. Validate the installed configuration parses and the embedding model is set.
"${CARACAL_PYTHON}" - <<PY
import os
os.environ["CARACAL_HOME"] = "${CARACAL_HOME}"
from caracal_rag.config import load_active_config, resolve_home
home = resolve_home()
config = load_active_config(home.config_path)
print(f"Active config: {home.config_path}")
print(f"Embedding model: {config.embedding.model}")
print(f"Embedding API base: {config.embedding.api_base}")
print(f"Chroma: {config.chroma.host}:{config.chroma.port}/{config.chroma.collection}")
print(f"Sources: {len(config.sources)}")
PY

echo
echo "Caracal RAG is installed at ${CARACAL_HOME}."
echo "Next steps:"
echo "  1. Set your FreeLLMAPI API key:  export CARACAL_EMBEDDING_API_KEY=<key>"
echo "     or run:                       caracal-rag setup --model bge-m3"
echo "  2. Validate connectivity:        caracal-rag check"
echo "  3. Build the index:              caracal-rag index"
echo "  4. Start the MCP server:         caracal-rag mcp"