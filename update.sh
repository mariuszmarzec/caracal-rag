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

# Install and validate into a new immutable generation. The active venv is not
# touched until candidate dependency/package installation and optional connectivity
# validation have all succeeded.
mkdir -p "$CARACAL_HOME"
CARACAL_HOME="$(cd "$CARACAL_HOME" && pwd -P)"
export CARACAL_HOME
ACTIVE_VENV="$CARACAL_HOME/venv"
GENERATIONS_DIR="$CARACAL_HOME/venvs"

if [[ -L "$GENERATIONS_DIR" || ( -e "$GENERATIONS_DIR" && ! -d "$GENERATIONS_DIR" ) ]]; then
    echo "ERROR: refusing an unsupported venv generations path: $GENERATIONS_DIR" >&2
    exit 1
fi
mkdir -p "$GENERATIONS_DIR"
[[ ! -L "$GENERATIONS_DIR" ]] || {
    echo "ERROR: refusing a symlink venv generations directory: $GENERATIONS_DIR" >&2
    exit 1
}
GENERATIONS_DIR="$(cd "$GENERATIONS_DIR" && pwd -P)"
chmod 0700 "$GENERATIONS_DIR"

ORIGINAL_KIND="missing"
OLD_VENV_TARGET=""
if [[ -L "$ACTIVE_VENV" ]]; then
    ORIGINAL_KIND="symlink"
    OLD_VENV_TARGET="$(readlink -f -- "$ACTIVE_VENV" 2>/dev/null || true)"
    [[ -n "$OLD_VENV_TARGET" && -d "$OLD_VENV_TARGET" ]] || {
        echo "ERROR: active venv symlink is broken: $ACTIVE_VENV" >&2
        echo "No update was attempted; repair the existing installation first." >&2
        exit 1
    }
elif [[ -d "$ACTIVE_VENV" ]]; then
    ORIGINAL_KIND="directory"
elif [[ -e "$ACTIVE_VENV" ]]; then
    echo "ERROR: active venv path is not a directory or managed symlink: $ACTIVE_VENV" >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CANDIDATE_VENV="$GENERATIONS_DIR/caracal-venv-$STAMP-$"
TEMP_LINK="$CARACAL_HOME/.venv-candidate-link-$"
LAUNCHER_TEMP=""
PROMOTION_STARTED=false
UPDATE_SUCCESS=false

restore_previous_active() {
    case "$ORIGINAL_KIND" in
        missing)
            if [[ -L "$ACTIVE_VENV" ]]; then
                local active_target candidate_target
                active_target="$(readlink -f -- "$ACTIVE_VENV" 2>/dev/null || true)"
                candidate_target="$(readlink -f -- "$CANDIDATE_VENV" 2>/dev/null || true)"
                if [[ -n "$candidate_target" && "$active_target" == "$candidate_target" ]]; then
                    rm -f -- "$ACTIVE_VENV"
                else
                    echo "ERROR: rollback refused to remove an active path it does not own: $ACTIVE_VENV" >&2
                    return 1
                fi
            elif [[ -e "$ACTIVE_VENV" ]]; then
                echo "ERROR: rollback found an unexpected active venv path: $ACTIVE_VENV" >&2
                return 1
            fi
            ;;
        symlink|directory)
            if [[ -n "$OLD_VENV_TARGET" && -d "$OLD_VENV_TARGET" ]]; then
                local rollback_link
                rollback_link="$CARACAL_HOME/.venv-rollback-$"
                if [[ -e "$rollback_link" || -L "$rollback_link" ]]; then
                    echo "ERROR: rollback link path already exists: $rollback_link" >&2
                    return 1
                fi
                if ! ln -s -- "$OLD_VENV_TARGET" "$rollback_link"; then
                    echo "ERROR: failed to prepare rollback symlink to $OLD_VENV_TARGET" >&2
                    return 1
                fi
                if ! mv -Tf -- "$rollback_link" "$ACTIVE_VENV"; then
                    rm -f -- "$rollback_link"
                    echo "ERROR: failed to restore the previous active venv symlink." >&2
                    return 1
                fi
            elif [[ "$ORIGINAL_KIND" == "directory" && -d "$ACTIVE_VENV" && ! -L "$ACTIVE_VENV" ]]; then
                # The legacy directory was never moved, or was already moved back
                # after a failed migration attempt. Preserve it in place.
                :
            else
                echo "ERROR: previous venv target is unavailable for rollback: ${OLD_VENV_TARGET:-<none>}" >&2
                return 1
            fi
            ;;
        *)
            echo "ERROR: unknown original venv state; refusing automatic rollback." >&2
            return 1
            ;;
    esac
}

cleanup_update() {
    local status=$?
    trap - EXIT

    if [[ "$UPDATE_SUCCESS" != true && "$PROMOTION_STARTED" == true ]]; then
        if restore_previous_active; then
            echo "Rollback restored the previous active venv; user configuration, secrets, manifest and Chroma were left untouched." >&2
        else
            echo "ERROR: automatic rollback failed. Do not delete either venv generation; inspect $ACTIVE_VENV and $GENERATIONS_DIR manually." >&2
        fi
    fi

    if [[ -n "$TEMP_LINK" && -L "$TEMP_LINK" ]]; then
        rm -f -- "$TEMP_LINK" || echo "WARNING: could not remove temporary symlink $TEMP_LINK" >&2
    fi
    if [[ -n "$LAUNCHER_TEMP" && -e "$LAUNCHER_TEMP" ]]; then
        rm -f -- "$LAUNCHER_TEMP" || echo "WARNING: could not remove temporary wrapper $LAUNCHER_TEMP" >&2
    fi

    if [[ -n "$CANDIDATE_VENV" && -d "$CANDIDATE_VENV" && ! -L "$CANDIDATE_VENV" ]]; then
        local active_target candidate_target
        active_target=""
        if [[ -L "$ACTIVE_VENV" ]]; then
            active_target="$(readlink -f -- "$ACTIVE_VENV" 2>/dev/null || true)"
        fi
        candidate_target="$(readlink -f -- "$CANDIDATE_VENV" 2>/dev/null || true)"
        if [[ -n "$candidate_target" && "$active_target" == "$candidate_target" ]]; then
            echo "WARNING: candidate remains active because rollback could not safely displace it: $CANDIDATE_VENV" >&2
        else
            rm -rf -- "$CANDIDATE_VENV" || echo "WARNING: could not remove failed candidate generation $CANDIDATE_VENV" >&2
        fi
    fi
    exit "$status"
}
trap cleanup_update EXIT

[[ ! -e "$CANDIDATE_VENV" && ! -L "$CANDIDATE_VENV" ]] || {
    echo "ERROR: candidate venv path already exists: $CANDIDATE_VENV" >&2
    exit 1
}
[[ ! -e "$TEMP_LINK" && ! -L "$TEMP_LINK" ]] || {
    echo "ERROR: temporary symlink path already exists: $TEMP_LINK" >&2
    exit 1
}

echo "Installing candidate Caracal RAG environment into $CANDIDATE_VENV..."
if CARACAL_VENV="$CANDIDATE_VENV" CARACAL_INSTALL_WRAPPER=false bash "$REPO_ROOT/install.sh"; then
    :
else
    install_status=$?
    echo "ERROR: candidate installation failed (exit $install_status); the active venv was not changed." >&2
    exit "$install_status"
fi

[[ -x "$CANDIDATE_VENV/bin/python" ]] || {
    echo "ERROR: candidate installation has no executable Python at $CANDIDATE_VENV/bin/python." >&2
    exit 1
}
[[ -x "$CANDIDATE_VENV/bin/caracal-rag" ]] || {
    echo "ERROR: candidate installation did not create $CANDIDATE_VENV/bin/caracal-rag." >&2
    echo "The previous active venv remains unchanged." >&2
    exit 1
}

if [[ "$RUN_CHECK" == true ]]; then
    echo "Validating the candidate environment before promotion..."
    if "$CANDIDATE_VENV/bin/caracal-rag" check; then
        :
    else
        check_status=$?
        echo "ERROR: candidate caracal-rag check failed (exit $check_status); the previous active venv remains unchanged." >&2
        exit "$check_status"
    fi
else
    echo "Skipping candidate connectivity check (--skip-check); install/entry-point validation still runs."
fi

# Prepare the candidate symlink before touching the active path.
ln -s -- "$CANDIDATE_VENV" "$TEMP_LINK"

if [[ "$ORIGINAL_KIND" == "directory" ]]; then
    OLD_VENV_TARGET="$GENERATIONS_DIR/caracal-venv-legacy-$STAMP-$"
    [[ ! -e "$OLD_VENV_TARGET" && ! -L "$OLD_VENV_TARGET" ]] || {
        echo "ERROR: legacy venv preservation path already exists: $OLD_VENV_TARGET" >&2
        exit 1
    }
fi

PROMOTION_STARTED=true
if [[ "$ORIGINAL_KIND" == "directory" ]]; then
    # Existing venv console-script shebangs still refer to ~/.caracal/venv/bin/python.
    # Move it to a retained generation and immediately keep that original path
    # resolving through a symlink, so rollback restores a runnable environment.
    if ! mv -- "$ACTIVE_VENV" "$OLD_VENV_TARGET"; then
        echo "ERROR: could not preserve the current venv generation; not promoting the candidate." >&2
        exit 1
    fi
    if ! ln -s -- "$OLD_VENV_TARGET" "$ACTIVE_VENV"; then
        if mv -- "$OLD_VENV_TARGET" "$ACTIVE_VENV"; then
            OLD_VENV_TARGET=""
        else
            echo "ERROR: failed to restore the legacy venv after creating its preservation symlink failed." >&2
        fi
        echo "ERROR: could not prepare rollback access to the legacy venv; not promoting the candidate." >&2
        exit 1
    fi
fi

if ! mv -Tf -- "$TEMP_LINK" "$ACTIVE_VENV"; then
    echo "ERROR: atomic venv promotion failed; attempting to restore the previous environment." >&2
    exit 1
fi
TEMP_LINK=""

# Older/manual installations may not have the stable wrapper. Create it only
# after candidate validation and promotion. Existing wrappers are not mutated.
if [[ ! -x "$CARACAL_HOME/bin/caracal-rag" ]]; then
    LAUNCHER_TEMP="$CARACAL_HOME/bin/.caracal-rag-update-$"
    if ! printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$ACTIVE_VENV/bin/caracal-rag" >"$LAUNCHER_TEMP"; then
        echo "ERROR: could not prepare the stable CLI wrapper." >&2
        exit 1
    fi
    chmod 0755 "$LAUNCHER_TEMP"
    if ! mv -f -- "$LAUNCHER_TEMP" "$CARACAL_HOME/bin/caracal-rag"; then
        echo "ERROR: could not install the stable CLI wrapper." >&2
        exit 1
    fi
    LAUNCHER_TEMP=""
fi

UPDATE_SUCCESS=true
echo
echo "Caracal RAG update/reinstall completed."
echo "Active venv: $CANDIDATE_VENV"
if [[ -n "$OLD_VENV_TARGET" && -d "$OLD_VENV_TARGET" ]]; then
    echo "Previous venv retained for rollback: $OLD_VENV_TARGET"
fi
echo "Configuration, secrets, index manifest and Chroma data were not rolled back or rebuilt."
echo "The update does not restart an already-running MCP process."
echo "Restart the process/host that launches Caracal RAG MCP to load the new code."
