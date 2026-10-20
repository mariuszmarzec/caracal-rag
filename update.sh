#!/usr/bin/env bash
# Update the source checkout and reinstall Caracal RAG into its isolated runtime.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash ./update.sh [--no-pull] [--skip-check]

Default: require a clean Git worktree, fast-forward pull the current branch,
reinstall from this checkout, and run "caracal-rag check".

Options:
  --no-pull     Reinstall from the current checkout without contacting Git.
                Useful for testing local changes; the worktree may be dirty.
  --skip-check  Skip the post-install connectivity/configuration check.
  -h, --help    Show this help.
EOF
}

DO_PULL=true
RUN_CHECK=true
while (($# > 0)); do
    case "$1" in
        --no-pull) DO_PULL=false ;;
        --skip-check) RUN_CHECK=false ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CARACAL_HOME="${CARACAL_HOME:-$HOME/.caracal}"
export CARACAL_HOME

command -v git >/dev/null 2>&1 || {
    echo "ERROR: Git is required to run this script." >&2
    exit 1
}
[[ -f "$REPO_ROOT/install.sh" && -f "$REPO_ROOT/pyproject.toml" ]] || {
    echo "ERROR: run this script from a Caracal RAG source checkout." >&2
    exit 1
}

GIT_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: $REPO_ROOT is not inside a Git working tree." >&2
    exit 1
}
GIT_ROOT="$(cd "$GIT_ROOT" && pwd -P)"
[[ "$GIT_ROOT" == "$REPO_ROOT" ]] || {
    echo "ERROR: update.sh must be run from the repository root: $GIT_ROOT" >&2
    exit 1
}

if [[ "$DO_PULL" == true ]]; then
    BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
    [[ -n "$BRANCH" ]] || {
        echo "ERROR: HEAD is detached; switch to a branch before updating." >&2
        exit 1
    }

    git -C "$REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1 || {
        echo "ERROR: branch '$BRANCH' has no configured upstream." >&2
        echo "Set one with: git branch --set-upstream-to=origin/$BRANCH $BRANCH" >&2
        exit 1
    }

    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
        echo "ERROR: the Git worktree is not clean. Commit or stash changes, then retry." >&2
        echo "To reinstall the current checkout without pulling, use: bash ./update.sh --no-pull" >&2
        git -C "$REPO_ROOT" status --short >&2
        exit 1
    fi

    echo "Updating branch '$BRANCH' with a fast-forward-only pull..."
    git -C "$REPO_ROOT" pull --ff-only
else
    echo "Skipping Git pull; using the current checkout as-is."
fi

echo "Reinstalling Caracal RAG into $CARACAL_HOME..."
bash "$REPO_ROOT/install.sh"

if [[ "$RUN_CHECK" == true ]]; then
    CLI="$CARACAL_HOME/bin/caracal-rag"
    [[ -x "$CLI" ]] || {
        echo "ERROR: installed CLI was not found at $CLI." >&2
        exit 1
    }
    echo "Running post-install validation..."
    "$CLI" check
else
    echo "Skipping post-install check (--skip-check)."
fi

echo
echo "Caracal RAG update/reinstall completed."
echo "The installer preserves your configuration, secrets and local manifest."
echo "It does not rebuild the index or restart an already-running MCP process."
echo "Restart the process/host that launches Caracal RAG MCP to load the new code."
