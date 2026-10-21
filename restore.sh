#!/usr/bin/env bash
# Restore Caracal local configuration, secrets, and optional manifest from backup.sh output.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash ./restore.sh BACKUP_DIR [--home DIR] [--force]

Restores the local files saved by backup.sh. Existing target files cause a
fail-closed refusal unless --force is supplied. With --force, the current
local files are first preserved in a separate private pre-restore snapshot.
This script never accesses, deletes, or restores remote Chroma vectors.

Options:
  --home DIR   Caracal state directory (default: $CARACAL_HOME or ~/.caracal)
  --force      Allow replacement; first save existing local files privately
  -h, --help   Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$SCRIPT_DIR"
STATE_HOME="${CARACAL_HOME:-$HOME/.caracal}"
FORCE=false
BACKUP_DIR=""

while (($# > 0)); do
    case "$1" in
        --home)
            [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --home needs a directory." >&2; exit 2; }
            STATE_HOME="$2"
            shift
            ;;
        --force) FORCE=true ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            [[ -z "$BACKUP_DIR" ]] || { echo "ERROR: provide exactly one backup directory." >&2; exit 2; }
            BACKUP_DIR="$1"
            ;;
    esac
    shift
done

[[ -n "$BACKUP_DIR" ]] || { usage >&2; exit 2; }
[[ -d "$BACKUP_DIR" && ! -L "$BACKUP_DIR" ]] || { echo "ERROR: backup directory not found or is a symlink." >&2; exit 1; }
BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd -P)"
BACKUP_INFO="$BACKUP_DIR/BACKUP-INFO.txt"
SOURCES_BACKUP="$BACKUP_DIR/config/sources.yaml"
SECRETS_BACKUP="$BACKUP_DIR/config/secrets.yaml"
MANIFEST_BACKUP="$BACKUP_DIR/index/manifest.json"

[[ -f "$BACKUP_INFO" && ! -L "$BACKUP_INFO" ]] || { echo "ERROR: BACKUP-INFO.txt is missing; this is not a recognized Caracal backup." >&2; exit 1; }
grep -qx "format=caracal-local-state-v1" "$BACKUP_INFO" || { echo "ERROR: unsupported backup format." >&2; exit 1; }
for required_file in "$SOURCES_BACKUP" "$SECRETS_BACKUP"; do
    if [[ ! -f "$required_file" || -L "$required_file" ]]; then
        echo "ERROR: backup is missing a regular required file: ${required_file#"$BACKUP_DIR"/}" >&2
        exit 1
    fi
done
[[ ! -L "$MANIFEST_BACKUP" ]] || { echo "ERROR: refusing a symlink manifest in the backup." >&2; exit 1; }
MANIFEST_STATUS="$(sed -n 's/^manifest_json=//p' "$BACKUP_INFO")"
if [[ "$MANIFEST_STATUS" == "included" && ! -f "$MANIFEST_BACKUP" ]]; then
    echo "ERROR: backup metadata says a manifest is included, but the file is missing." >&2
    exit 1
fi
if [[ "$MANIFEST_STATUS" == "absent" && -e "$MANIFEST_BACKUP" ]]; then
    echo "ERROR: backup metadata says the manifest is absent, but a manifest file exists." >&2
    exit 1
fi
[[ "$MANIFEST_STATUS" == "included" || "$MANIFEST_STATUS" == "absent" ]] || {
    echo "ERROR: backup manifest status is missing or invalid." >&2
    exit 1
}

if [[ -L "$STATE_HOME" ]]; then echo "ERROR: refusing a symlink Caracal state directory." >&2; exit 1; fi
mkdir -p "$STATE_HOME"
STATE_HOME="$(cd "$STATE_HOME" && pwd -P)"
TARGET_SOURCES="$STATE_HOME/config/sources.yaml"
TARGET_SECRETS="$STATE_HOME/config/secrets.yaml"
TARGET_MANIFEST="$STATE_HOME/index/manifest.json"

for target in "$TARGET_SOURCES" "$TARGET_SECRETS" "$TARGET_MANIFEST"; do
    [[ ! -L "$target" ]] || { echo "ERROR: refusing to overwrite a symlink: $target" >&2; exit 1; }
done
NEEDS_REPLACEMENT=false
for target in "$TARGET_SOURCES" "$TARGET_SECRETS"; do [[ ! -e "$target" ]] || NEEDS_REPLACEMENT=true; done
if [[ -e "$TARGET_MANIFEST" || "$MANIFEST_STATUS" == "included" ]]; then
    [[ ! -e "$TARGET_MANIFEST" && "$MANIFEST_STATUS" == "absent" ]] || NEEDS_REPLACEMENT=true
fi
if [[ "$NEEDS_REPLACEMENT" == true && "$FORCE" != true ]]; then
    echo "ERROR: target state already contains files that would change." >&2
    echo "No files were restored. Inspect the target or repeat with --force to preserve a safety snapshot first." >&2
    exit 1
fi

# Prepare every restored file before changing any active target.
STAGING="$(mktemp -d "$STATE_HOME/.restore-stage.XXXXXX")"
cleanup() { if [[ -n "${STAGING:-}" && -d "$STAGING" ]]; then rm -rf -- "$STAGING"; fi; }
trap cleanup EXIT
chmod 0700 "$STAGING"
mkdir -p "$STAGING/config" "$STAGING/index"
cp -- "$SOURCES_BACKUP" "$STAGING/config/sources.yaml"
cp -- "$SECRETS_BACKUP" "$STAGING/config/secrets.yaml"
chmod 0600 "$STAGING/config/sources.yaml" "$STAGING/config/secrets.yaml"
if [[ "$MANIFEST_STATUS" == "included" ]]; then
    cp -- "$MANIFEST_BACKUP" "$STAGING/index/manifest.json"
    chmod 0600 "$STAGING/index/manifest.json"
fi

PRESERVE_DIR=""
if [[ "$FORCE" == true && "$NEEDS_REPLACEMENT" == true ]]; then
    PRESERVE_ROOT="${CARACAL_BACKUP_ROOT:-$HOME/caracal-backups}"
    mkdir -p "$PRESERVE_ROOT"
    PRESERVE_ROOT="$(cd "$PRESERVE_ROOT" && pwd -P)"
    PRESERVE_DIR="$PRESERVE_ROOT/pre-restore-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    case "$PRESERVE_DIR" in
        "$REPO_ROOT"|"$REPO_ROOT"/*) echo "ERROR: pre-restore safety copy cannot be inside the source repository." >&2; exit 1 ;;
    esac
    PRESERVE_STAGE="$(mktemp -d "$PRESERVE_ROOT/.pre-restore.XXXXXX")"
    chmod 0700 "$PRESERVE_STAGE"
    mkdir -p "$PRESERVE_STAGE/config" "$PRESERVE_STAGE/index"
    for pair in "$TARGET_SOURCES:config/sources.yaml" "$TARGET_SECRETS:config/secrets.yaml" "$TARGET_MANIFEST:index/manifest.json"; do
        source_path="${pair%%:*}"
        relative_path="${pair#*:}"
        if [[ -f "$source_path" ]]; then
            cp -- "$source_path" "$PRESERVE_STAGE/$relative_path"
            chmod 0600 "$PRESERVE_STAGE/$relative_path"
        fi
    done
    cat >"$PRESERVE_STAGE/BACKUP-INFO.txt" <<EOF
format=caracal-pre-restore-safety-v1
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
purpose=Safety snapshot created by restore.sh before replacing local files
chroma_vectors=included:NO
EOF
    chmod 0600 "$PRESERVE_STAGE/BACKUP-INFO.txt"
    mv -- "$PRESERVE_STAGE" "$PRESERVE_DIR"
fi

mkdir -p "$STATE_HOME/config" "$STATE_HOME/index"
mv -- "$STAGING/config/sources.yaml" "$TARGET_SOURCES"
mv -- "$STAGING/config/secrets.yaml" "$TARGET_SECRETS"
if [[ "$MANIFEST_STATUS" == "included" ]]; then
    mv -- "$STAGING/index/manifest.json" "$TARGET_MANIFEST"
elif [[ "$FORCE" == true && -e "$TARGET_MANIFEST" ]]; then
    # Faithfully restore a backup that predates indexing. The previous manifest
    # has been preserved in the pre-restore safety snapshot above.
    rm -- "$TARGET_MANIFEST"
fi
chmod 0700 "$STATE_HOME" "$STATE_HOME/config" "$STATE_HOME/index"
chmod 0600 "$TARGET_SOURCES" "$TARGET_SECRETS"
[[ ! -f "$TARGET_MANIFEST" ]] || chmod 0600 "$TARGET_MANIFEST"

STAGING=""
trap - EXIT
echo "Caracal local state restored to: $STATE_HOME"
if [[ -n "$PRESERVE_DIR" ]]; then echo "Previous local files preserved at: $PRESERVE_DIR"; fi
echo "Remote Chroma vectors were not restored. Verify that the configured Chroma collection matches this backup before serving searches."
echo "Next: run the installed caracal-rag check command, then a real authenticated MCP search smoke test."
