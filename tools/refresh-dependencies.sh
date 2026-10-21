#!/usr/bin/env bash
# Generate/check the per-minor Python dependency constraints used by install.sh and the CI matrix.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PIP_TOOLS_VERSION="7.6.2"
SUPPORTED_PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")

MODE="refresh"
TARGET_PYTHON=""
UPGRADE_PACKAGE=""

usage() {
    cat <<'EOF'
Usage: bash tools/refresh-dependencies.sh [--check | --upgrade | --upgrade-package NAME] [--python VERSION]

Without an upgrade option, retain compatible existing pins while resolving
new/changed dependencies. --upgrade refreshes all pins; --upgrade-package
refreshes one package and the compatible dependency graph around it.
--check verifies the checked-in lock(s) without modifying them.
--python VERSION limits the operation to one supported Python minor version.
Supported Python minors: 3.10, 3.11, 3.12, 3.13, 3.14.
EOF
}

while (($# > 0)); do
    case "$1" in
        --check|--upgrade)
            if [[ "$MODE" != "refresh" ]]; then
                echo "ERROR: choose only one of --check, --upgrade, or --upgrade-package." >&2
                exit 2
            fi
            MODE="${1#--}"
            ;;
        --upgrade-package)
            if [[ "$MODE" != "refresh" || $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --upgrade-package requires one package name and cannot be combined with another mode." >&2
                exit 2
            fi
            MODE="upgrade-package"
            UPGRADE_PACKAGE="$2"
            shift
            ;;
        --python)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "ERROR: --python requires a version such as 3.12." >&2
                exit 2
            fi
            TARGET_PYTHON="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ -n "$TARGET_PYTHON" ]]; then
    supported=false
    for version in "${SUPPORTED_PYTHON_VERSIONS[@]}"; do
        [[ "$version" == "$TARGET_PYTHON" ]] && supported=true
    done
    if [[ "$supported" != true ]]; then
        echo "ERROR: unsupported Python version '$TARGET_PYTHON'." >&2
        usage >&2
        exit 2
    fi
    PYTHON_VERSIONS=("$TARGET_PYTHON")
else
    PYTHON_VERSIONS=("${SUPPORTED_PYTHON_VERSIONS[@]}")
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

cd "$REPO_ROOT"
for version in "${PYTHON_VERSIONS[@]}"; do
    tag="${version//./}"
    interpreter="python${version}"
    lock_file="$REPO_ROOT/constraints-py${tag}.lock"
    if ! command -v "$interpreter" >/dev/null 2>&1; then
        echo "ERROR: $interpreter is required to manage the complete lock matrix." >&2
        exit 1
    fi

    tool_venv="$TMP_ROOT/pip-tools-py${tag}"
    "$interpreter" -m venv "$tool_venv"
    tool_python="$tool_venv/bin/python"
    "$tool_python" -m pip install --disable-pip-version-check "pip-tools==${PIP_TOOLS_VERSION}"

    compile_args=(
        --no-config
        --all-build-deps
        --extra dev
        --strip-extras
        --no-header
        --no-annotate
        --no-emit-index-url
        --allow-unsafe
        --resolver=backtracking
    )

    if [[ "$MODE" == "check" ]]; then
        if [[ ! -f "$lock_file" ]]; then
            echo "ERROR: lock file is missing: $lock_file" >&2
            exit 1
        fi
        output_file="$TMP_ROOT/constraints-py${tag}.lock"
        compile_args+=(--constraint "$lock_file" --output-file "$output_file")
    else
        output_file="$lock_file"
        compile_args+=(--output-file "$output_file")
        if [[ "$MODE" == "upgrade" ]]; then
            compile_args+=(--upgrade)
        elif [[ "$MODE" == "upgrade-package" ]]; then
            compile_args+=(--upgrade-package "$UPGRADE_PACKAGE")
        fi
    fi

    echo "Resolving dependency pins for Python $version..."
    "$tool_python" -m piptools compile "${compile_args[@]}" pyproject.toml

    if [[ "$MODE" == "check" ]] && ! cmp -s "$lock_file" "$output_file"; then
        echo "ERROR: $lock_file is out of sync with pyproject.toml." >&2
        diff -u "$lock_file" "$output_file" || true
        exit 1
    fi
done

if [[ "$MODE" == "check" ]]; then
    echo "All checked dependency locks match pyproject.toml."
else
    echo "Dependency lock files refreshed. Review the diff and run the full CI matrix before merging."
fi
