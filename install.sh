#!/usr/bin/env bash
# Install Caracal RAG into an isolated environment under ~/.caracal.
# Safe to rerun: active config, secrets, and an existing index are preserved.
set -euo pipefail

CARACAL_HOME="${CARACAL_HOME:-$HOME/.caracal}"
CARACAL_PYTHON="${CARACAL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_CONFIG="${REPO_ROOT}/config/sources.example.yaml"

mkdir -p "$CARACAL_HOME"
CARACAL_HOME="$(cd "$CARACAL_HOME" && pwd -P)"
CONFIG_DIR="$CARACAL_HOME/config"
INDEX_DIR="$CARACAL_HOME/index"
BIN_DIR="$CARACAL_HOME/bin"
VENV="$CARACAL_HOME/venv"
CONFIG_PATH="$CONFIG_DIR/sources.yaml"
SECRETS_PATH="$CONFIG_DIR/secrets.yaml"
MANIFEST_PATH="$INDEX_DIR/manifest.json"

echo "Installing Caracal RAG into $CARACAL_HOME"
mkdir -p "$CONFIG_DIR" "$INDEX_DIR" "$BIN_DIR"

command -v "$CARACAL_PYTHON" >/dev/null 2>&1 || {
    echo "ERROR: Python interpreter '$CARACAL_PYTHON' was not found." >&2
    exit 1
}

# The runtime must not depend on a system-wide pip install or on this checkout.
# Keep the venv under CARACAL_HOME and install the package non-editably.
if [ ! -x "$VENV/bin/python" ]; then
    "$CARACAL_PYTHON" -m venv "$VENV"
fi
[ -x "$VENV/bin/python" ] || {
    echo "ERROR: virtual environment was not created at $VENV." >&2
    exit 1
}
# Select the lock for the interpreter that actually runs the installed environment.
# This also handles updates where CARACAL_PYTHON differs from an existing venv.
PYTHON_VERSION="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_TAG="${PYTHON_VERSION//./}"
CONSTRAINTS_FILE="${REPO_ROOT}/constraints-py${PYTHON_TAG}.lock"
if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
    echo "ERROR: no locked dependency set is available for Python $PYTHON_VERSION." >&2
    echo "Supported interpreter minors: 3.10, 3.11, 3.12, 3.13, 3.14." >&2
    echo "Add and verify constraints-py${PYTHON_TAG}.lock before installing with this Python." >&2
    exit 1
fi
# PIP_CONSTRAINT is inherited by PEP 517 build-isolation subprocesses; -c
# also constrains the main install resolution. Keep build/runtime/test versions aligned.
PIP_CONSTRAINT="$CONSTRAINTS_FILE" "$VENV/bin/python" -m pip install --constraint "$CONSTRAINTS_FILE" --upgrade --force-reinstall "$REPO_ROOT"

# Seed config only on first install. Never repair/replace an existing config
# automatically: a malformed file may contain valuable user-defined settings.
if [ ! -e "$CONFIG_PATH" ]; then
    if [ -f "$EXAMPLE_CONFIG" ]; then
        cp "$EXAMPLE_CONFIG" "$CONFIG_PATH"
        echo "Wrote $CONFIG_PATH from the example template"
    else
        cat >"$CONFIG_PATH" <<'EOF'
embedding:
  model: bge-m3
  api_base: http://localhost:3001/v1

chroma:
  host: localhost
  port: 3400
  collection: caracal-base

sources: []
EOF
        echo "Wrote $CONFIG_PATH from the built-in template"
    fi
else
    echo "Preserving existing $CONFIG_PATH"
fi

# Secrets always live outside the source checkout and stay private.
if [ ! -e "$SECRETS_PATH" ]; then
    : >"$SECRETS_PATH"
    chmod 0600 "$SECRETS_PATH"
    echo "Created $SECRETS_PATH (mode 0600; configure it with 'caracal-rag setup')"
else
    [ -f "$SECRETS_PATH" ] || {
        echo "ERROR: expected a regular secrets file at $SECRETS_PATH." >&2
        exit 1
    }
    chmod 0600 "$SECRETS_PATH"
    echo "Preserving existing secrets in $SECRETS_PATH (mode ensured 0600)"
fi

if [ -f "$MANIFEST_PATH" ]; then
    echo "Preserving existing index manifest $MANIFEST_PATH"
fi

# Stable wrapper at ~/.caracal/bin/caracal-rag; no global install or symlink
# into the source checkout. It dispatches to the venv installed entry point.
[ -x "$VENV/bin/caracal-rag" ] || {
    echo "ERROR: pip install completed without creating $VENV/bin/caracal-rag." >&2
    exit 1
}
printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$VENV/bin/caracal-rag" >"$BIN_DIR/caracal-rag"
chmod 0755 "$BIN_DIR/caracal-rag"

CARACAL_HOME="$CARACAL_HOME" "$VENV/bin/python" - <<'PY'
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
echo "Caracal RAG is installed at $CARACAL_HOME."
echo "Add the CLI directory to PATH if it is not already available:"
echo "  export PATH=\"$CARACAL_HOME/bin:\$PATH\""
echo "Next steps:"
echo "  1. Configure the model, key and access token: caracal-rag setup --model bge-m3"
echo "     (the key and token are prompted without echo; alternatively export"
echo "      CARACAL_EMBEDDING_API_KEY and CARACAL_ACCESS_TOKEN)"
echo "  2. Validate connectivity:       caracal-rag check"
echo "  3. Build the index:             caracal-rag index"
echo "  4. Start MCP:                   caracal-rag mcp"
