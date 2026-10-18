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
"$VENV/bin/python" -m pip install --upgrade --force-reinstall "$REPO_ROOT"

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
echo "  1. Configure the model and key: caracal-rag setup --model bge-m3"
echo "     (the key is prompted without echo; alternatively export CARACAL_EMBEDDING_API_KEY)"
echo "  2. Validate connectivity:       caracal-rag check"
echo "  3. Build the index:             caracal-rag index"
echo "  4. Start MCP:                   caracal-rag mcp"
