#!/usr/bin/env bash
# Back up Caracal local configuration, secrets, and index manifest only.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash ./backup.sh [--home DIR] [--output DIR]

Backs up config/sources.yaml, config/secrets.yaml, and index/manifest.json
(when present). Default output is a timestamped folder under
$HOME/caracal-backups. Backup directories are private (0700); files are 0600.
Remote Chroma vectors are NOT included.

Options:
  --home DIR     Caracal state directory (default: $CARACAL_HOME or ~/.caracal)
  --output DIR   Exact new destination directory (must not already exist)
  -h, --help     Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$SCRIPT_DIR"
STATE_HOME="${CARACAL_HOME:-$HOME/.caracal}"
OUTPUT=""

while (($# > 0)); do
    case "$1" in
        --home)
            [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --home needs a directory." >&2; exit 2; }
            STATE_HOME="$2"
            shift
            ;;
        --output)
            [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --output needs a directory." >&2; exit 2; }
            OUTPUT="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ -d "$STATE_HOME" ]] || { echo "ERROR: Caracal state directory not found: $STATE_HOME" >&2; exit 1; }
STATE_HOME="$(cd "$STATE_HOME" && pwd -P)"
SOURCES_FILE="$STATE_HOME/config/sources.yaml"
SECRETS_FILE="$STATE_HOME/config/secrets.yaml"
MANIFEST_FILE="$STATE_HOME/index/manifest.json"

for directory in "$STATE_HOME/config" "$STATE_HOME/index"; do
    if [[ -L "$directory" ]]; then
        echo "ERROR: refusing a symlink local-state directory: $directory" >&2
        exit 1
    fi
done

for required_file in "$SOURCES_FILE" "$SECRETS_FILE"; do
    if [[ ! -f "$required_file" || -L "$required_file" ]]; then
        echo "ERROR: expected a regular local state file at $required_file." >&2
        echo "Run the Caracal installer/setup first, then retry the backup." >&2
        exit 1
    fi
done
if [[ -L "$MANIFEST_FILE" ]]; then
    echo "ERROR: refusing to follow a symlink at $MANIFEST_FILE." >&2
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${CARACAL_BACKUP_ROOT:-$HOME/caracal-backups}/caracal-$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$PWD/$OUTPUT"
fi
OUT_PARENT="$(dirname "$OUTPUT")"
OUT_NAME="$(basename "$OUTPUT")"
[[ -n "$OUT_NAME" && "$OUT_NAME" != "." && "$OUT_NAME" != ".." ]] || {
    echo "ERROR: invalid output directory." >&2
    exit 2
}
mkdir -p "$OUT_PARENT"
OUT_PARENT="$(cd "$OUT_PARENT" && pwd -P)"
OUTPUT="$OUT_PARENT/$OUT_NAME"
case "$OUTPUT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
        echo "ERROR: backups must not be created inside the source repository." >&2
        echo "Choose a directory outside $REPO_ROOT." >&2
        exit 1
        ;;
esac
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: output already exists: $OUTPUT" >&2; exit 1; }

STAGING="$(mktemp -d "$OUT_PARENT/.caracal-backup.XXXXXX")"
cleanup() {
    if [[ -n "${STAGING:-}" && -d "$STAGING" ]]; then rm -rf -- "$STAGING"; fi
}
trap cleanup EXIT
chmod 0700 "$STAGING"
mkdir -p "$STAGING/config" "$STAGING/index"
cp -- "$SOURCES_FILE" "$STAGING/config/sources.yaml"
cp -- "$SECRETS_FILE" "$STAGING/config/secrets.yaml"
chmod 0600 "$STAGING/config/sources.yaml" "$STAGING/config/secrets.yaml"

MANIFEST_STATUS="absent"
if [[ -f "$MANIFEST_FILE" ]]; then
    cp -- "$MANIFEST_FILE" "$STAGING/index/manifest.json"
    chmod 0600 "$STAGING/index/manifest.json"
    MANIFEST_STATUS="included"
fi
cat >"$STAGING/BACKUP-INFO.txt" <<EOF
format=caracal-local-state-v1
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sources_yaml=included
secrets_yaml=included
manifest_json=$MANIFEST_STATUS
chroma_vectors=included:NO
chroma_note=Remote Chroma data must be backed up separately using the actual Chroma deployment storage/backup procedure.
EOF
chmod 0600 "$STAGING/BACKUP-INFO.txt"
mv -- "$STAGING" "$OUTPUT"
STAGING=""
trap - EXIT

echo "Caracal local-state backup created: $OUTPUT"
echo "Included sources.yaml and secrets.yaml; manifest: $MANIFEST_STATUS."
echo "Remote Chroma vectors were not copied. No secret values were printed."
