"""Automated safety tests for ``update.sh`` (issue #21).

These tests exercise the update/reinstall workflow in *isolated temporary Git
repositories* with fake Git/install/CLI commands where practical. They never
touch the real user's ``~/.caracal`` or the actual checkout: every Git
repository is created under ``tmp_path``, and every subprocess is run with
``HOME`` pointed at a temporary directory.

Covered required scenarios
--------------------------
* A dirty worktree (modified, staged, or untracked files) blocks the default
  update before any pull or install.
* Detached HEAD and a branch without an upstream fail with actionable errors
  before installation.
* A pull that cannot fast-forward fails and does not reinstall.
* ``--no-pull`` installs the current checkout without attempting a Git network
  operation, including when the worktree is dirty.
* Installer failure propagates a non-zero exit and does not run the
  post-install check.
* Post-install ``caracal-rag check`` failure propagates a non-zero exit.
* ``--skip-check`` skips only the post-install check.
* ``CARACAL_HOME`` is passed through consistently.
* Unknown arguments fail with exit code 2 and useful usage text.

The suite is deterministic and needs no network access or real API
credentials: the fake Git commands are plain bash scripts that log their
invocation instead of contacting a remote, and the "pullable" scenarios are
created with a local bare remote plus detached worktrees (no network).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATE_SH = REPO_ROOT / "update.sh"

# Real user home must remain untouched by every test.
REAL_HOME = Path(os.path.expanduser("~"))
REAL_CARACAL_HOME = REAL_HOME / ".caracal"

# Log files the fakes write. They double as an "operation happened" oracle.
PULL_LOG_NAME = "pull.log"
INSTALL_LOG_NAME = "install.log"
CHECK_LOG_NAME = "check.log"
CANDIDATE_REMAINS_ACTIVE_WARNING = (
    "WARNING: candidate remains active because rollback could not safely displace it"
)


def _run(
    args: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git(repo: Path, *args: str, expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in *repo*, treating a non-zero exit as a test error
    unless ``expect_ok`` is False."""
    proc = _run(["git", "-C", str(repo), *args])
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"git {args} failed (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}"
        )
    return proc


@dataclass
class FakeEnv:
    """A self-contained sandbox: a temporary Git repo, a fake ``update.sh``, a
    fake installer, and a fake ``caracal-rag`` CLI. Everything lives under
    ``tmp_path`` so no test can touch the real home or checkout."""

    work: Path
    repo: Path
    caracal_home: Path
    fake_cli: Path
    git_env: dict[str, str]
    base_env: dict[str, str]

    @property
    def pull_log(self) -> Path:
        return self.work / PULL_LOG_NAME

    @property
    def install_log(self) -> Path:
        return self.work / INSTALL_LOG_NAME

    @property
    def check_log(self) -> Path:
        return self.work / CHECK_LOG_NAME

    @property
    def logs(self) -> list[Path]:
        return [self.pull_log, self.install_log, self.check_log]

    def _fresh(self) -> None:
        """Reset the sandbox to a clean state and drop any per-run flags."""
        for log in self.logs:
            if log.exists():
                log.unlink()
        self.base_env.pop("INSTALL_FAIL", None)
        self.base_env.pop("CHECK_FAIL", None)

    def make_dirty(self, kind: str) -> None:
        """Dirty the working tree: ``modified``, ``staged`` or ``untracked``."""
        if kind == "modified":
            (self.repo / "README.md").write_text("touched by test\n", encoding="utf-8")
        elif kind == "staged":
            staged = self.repo / "staged.txt"
            staged.write_text("staged\n", encoding="utf-8")
            _git(self.repo, "add", "staged.txt")
        elif kind == "untracked":
            (self.repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        else:
            raise ValueError(f"unknown dirty kind: {kind}")

    def detach_head(self) -> None:
        _git(self.repo, "checkout", "--detach")
        check = _git(self.repo, "symbolic-ref", "HEAD", expect_ok=False)
        assert check.returncode != 0, "HEAD is unexpectedly still a symbolic ref"

    def detach_head_at(self, ref: str) -> None:
        _git(self.repo, "checkout", "--detach", ref)

    def new_branch(self, name: str) -> None:
        _git(self.repo, "checkout", "-b", name)

    def head(self) -> str:
        return _git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def run_update(
        self,
        *args: str,
        extra_env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        self._fresh()
        env = dict(self.base_env)
        if extra_env:
            env.update(extra_env)
        return _run(["bash", str(self.repo / "update.sh"), *args], cwd=self.repo, env=env)

    def assert_not_ran(self, *paths: Path) -> None:
        for p in paths:
            assert not p.exists(), f"expected {p.name} to be absent, but it exists: {p.read_text()}"


def _fake_install_sh() -> str:
    # A deterministic candidate installer. Failure modes can leave a partial
    # generation behind, which update.sh must remove without touching the active venv.
    prefix = """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'install invoked; CARACAL_HOME=%s; CARACAL_VENV=%s; CARACAL_INSTALL_WRAPPER=%s\\n' \\
    "${CARACAL_HOME:-<unset>}" \\
    "${CARACAL_VENV:-<unset>}" \\
    "${CARACAL_INSTALL_WRAPPER:-<unset>}" >> "$INSTALL_LOG"
VENV="${CARACAL_VENV:?CARACAL_VENV must name the candidate environment}"
mkdir -p "$VENV/bin"
cat >"$VENV/bin/python" <<'PYTHON'
#!/usr/bin/env bash
exit 0
PYTHON
chmod 0755 "$VENV/bin/python"
if [[ "${INSTALL_FAIL_STAGE:-}" == "dependencies" ]]; then
    echo "FAKE dependency installation failure (exit 5)" >&2
    exit 5
fi
if [[ "${MISSING_ENTRYPOINT:-0}" != "1" ]]; then
    cat >"$VENV/bin/caracal-rag" <<'CLI'
"""
    suffix = """CLI
    chmod 0755 "$VENV/bin/caracal-rag"
fi
if [[ "${INSTALL_FAIL_STAGE:-}" == "package" ]]; then
    echo "FAKE package installation failure (exit 6)" >&2
    exit 6
fi
if [[ "${INSTALL_FAIL:-0}" == "1" ]]; then
    echo "FAKE installer failure (exit 5)" >&2
    exit 5
fi
exit 0
"""
    return prefix + _fake_cli_script() + suffix


def _fake_cli_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        '# FAKE caracal-rag CLI: logs every invocation (proving the post-install\n'
        '# check ran and what args/env it got) and honours an optional failure\n'
        '# flag so we can test that a failed `check` propagates.\n'
        'echo "cli invoked; args=$*; CARACAL_HOME=${CARACAL_HOME:-<unset>}" >> "$CHECK_LOG"\n'
        'if [[ "${1:-}" == "check" ]]; then\n'
        '    if [[ "${CHECK_FAIL:-0}" == "1" ]]; then\n'
        '        echo "FAKE check failure (exit 9)" >&2\n'
        '        exit 9\n'
        '    fi\n'
        '    echo "check ok"\n'
        'fi\n'
        'exit 0\n'
    )


def _fake_launcher_script() -> str:
    """Stable CLI shim that always dispatches through the active venv path."""
    return (
        "#!/usr/bin/env bash\n"
        'exec "$CARACAL_HOME/venv/bin/caracal-rag" "$@"\n'
    )


def _fake_git_wrapper(real_git: str) -> str:
    """A ``git`` PATH shim that logs any ``git pull`` invocation and then
    delegates to the real Git binary.

    The real ``update.sh`` calls ``git pull --ff-only`` directly, so the only
    way to observe whether a network Git operation was attempted is to
    intercept ``git`` itself. The wrapper is a thin passthrough: every other
    ``git`` command (init, add, commit, push, worktree, rev-parse, ...) runs
    unchanged, so the fake repos behave like real Git repositories and the
    fast-forward / non-fast-forward outcomes are genuine.
    """
    return (
        "#!/usr/bin/env bash\n"
        "# FAKE git wrapper: logs any `git pull` and delegates to real git.\n"
        'if [[ " $* " == *" pull "* ]]; then\n'
        '    echo "pull attempted" >> "$PULL_LOG"\n'
        'fi\n'
        f'exec "{real_git}" "$@"\n'
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _prepare(tmp_path: Path) -> FakeEnv:
    """Build the sandbox and return it ready to run the real ``update.sh``."""
    work = tmp_path
    repo = work / "fake"
    caracal_home = work / "home"
    remote = work / "remote.git"

    work.mkdir(parents=True, exist_ok=True)
    repo.mkdir()
    caracal_home.mkdir()
    (caracal_home / "bin").mkdir()
    fake_cli = caracal_home / "bin" / "caracal-rag"
    active_venv = caracal_home / "venv"
    (active_venv / "bin").mkdir(parents=True)
    _write_executable(active_venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(active_venv / "bin" / "caracal-rag", _fake_cli_script())

    shutil.copy(UPDATE_SH, repo / "update.sh")
    _write_executable(repo / "update.sh", (repo / "update.sh").read_text(encoding="utf-8"))
    (repo / "pyproject.toml").write_text('name = "caracal-rag"\n', encoding="utf-8")
    (repo / "README.md").write_text("fake caracal-rag checkout\n", encoding="utf-8")
    _write_executable(repo / "install.sh", _fake_install_sh())
    _write_executable(fake_cli, _fake_launcher_script())
    # A `git` PATH shim (named `git`, in a directory outside the repo so it
    # does not pollute the working tree) that logs any `git pull` invocation
    # and delegates to the real Git binary. The real `update.sh` calls
    # `git pull --ff-only` directly, so the only way to observe whether a
    # network Git operation was attempted is to intercept `git` itself. The
    # wrapper is a thin passthrough: every other `git` command (init, add,
    # commit, push, worktree, rev-parse, ...) runs unchanged, so the fake
    # repos behave like real Git repositories and the fast-forward /
    # non-fast-forward outcomes are genuine.
    real_git = shutil.which("git")
    assert real_git, "git binary must be on PATH for the test sandbox"
    fake_bin = work / "fakebin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "git", _fake_git_wrapper(real_git))

    # Local bare remote + initial push, so fast-forward pulls work offline.
    _git(repo, "-c", "init.defaultBranch=master", "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(remote.parent, "-c", "init.defaultBranch=master", "init", "--bare", remote.name)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "master")
    assert _git(repo, "status", "--porcelain").stdout == "", "fresh fake repo must be clean"

    env = dict(os.environ)
    env["HOME"] = str(work)
    env["CARACAL_HOME"] = str(caracal_home)
    env["PULL_LOG"] = str(work / PULL_LOG_NAME)
    env["INSTALL_LOG"] = str(work / INSTALL_LOG_NAME)
    env["CHECK_LOG"] = str(work / CHECK_LOG_NAME)
    # Isolate git from any global/system config so behaviour is deterministic.
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_AUTHOR_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    # The fake `git` wrapper must shadow the real binary when update.sh runs.
    env["PATH"] = f"{fake_bin}:{os.environ.get('PATH', '')}"
    # Drop any stray flags so the baseline run is clean.
    env.pop("INSTALL_FAIL", None)
    env.pop("CHECK_FAIL", None)

    return FakeEnv(
        work=work,
        repo=repo,
        caracal_home=caracal_home,
        fake_cli=fake_cli,
        git_env={},
        base_env=env,
    )


def _assert_existing_active_venv_works(env: FakeEnv) -> None:
    """The old installed CLI remains runnable after an aborted candidate update."""
    active = env.caracal_home / "venv"
    assert active.is_dir()
    assert (active / "bin" / "caracal-rag").is_file()
    proc = _run(
        [str(env.fake_cli), "check"],
        cwd=env.repo,
        env=dict(env.base_env),
    )
    assert proc.returncode == 0, proc.stderr
    assert "check ok" in proc.stdout


def _assert_no_candidate_generations(env: FakeEnv) -> None:
    generations = env.caracal_home / "venvs"
    if generations.exists():
        leftovers = list(generations.glob("caracal-venv-*"))
        assert leftovers == [], f"failed candidate generations were not cleaned: {leftovers}"


def test_dirty_worktree_blocks_default_update(tmp_path: Path) -> None:
    """A dirty worktree (modified, staged, or untracked) blocks the default
    update before any pull or install."""
    for kind in ("modified", "staged", "untracked"):
        env = _prepare(tmp_path / kind)
        env.make_dirty(kind)
        proc = env.run_update()
        assert proc.returncode != 0, f"{kind}: expected non-zero exit, got {proc.returncode}"
        clean_msg = re.search(r"not clean|worktree", proc.stderr, re.IGNORECASE)
        assert clean_msg, (
            f"{kind}: expected an actionable clean-worktree error; "
            f"got stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        env.assert_not_ran(env.install_log, env.check_log)


def test_detached_head_fails_before_install(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    env.detach_head()
    proc = env.run_update()
    assert proc.returncode != 0
    assert "HEAD is detached" in proc.stderr
    env.assert_not_ran(env.pull_log, env.install_log, env.check_log)


def test_branch_without_upstream_fails_before_install(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    env.new_branch("local")
    proc = env.run_update()
    assert proc.returncode != 0
    assert "no configured upstream" in proc.stderr
    env.assert_not_ran(env.pull_log, env.install_log, env.check_log)


def test_non_ff_pull_fails_and_does_not_reinstall(tmp_path: Path) -> None:
    env = _prepare(tmp_path)

    # Commit an extra commit on the remote so the local branch cannot
    # fast-forward to it.
    _git(env.repo, "commit", "--allow-empty", "-m", "local extra")
    wt = env.work / "wt"
    _git(env.repo, "worktree", "add", "--detach", str(wt), "origin/master")
    (wt / "remote-file.txt").write_text("from remote\n", encoding="utf-8")
    _git(wt, "add", "remote-file.txt")
    _git(wt, "commit", "-m", "remote extra")
    _git(wt, "push", "-f", "origin", "HEAD:refs/heads/master")
    _git(env.repo, "worktree", "remove", "--force", str(wt))

    proc = env.run_update()
    assert proc.returncode != 0, (
        f"expected non-ff pull to fail; got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    # The pull was attempted but the fast-forward was rejected, so no
    # reinstall happened.
    assert env.pull_log.exists(), "expected the pull to be attempted"
    env.assert_not_ran(env.install_log, env.check_log)


def test_no_pull_installs_current_checkout_without_network(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    before = env.head()
    proc = env.run_update("--no-pull")
    assert proc.returncode == 0, proc.stderr
    # No Git network operation was attempted.
    env.assert_not_ran(env.pull_log)
    # The current checkout was reinstalled and the post-install check ran.
    assert env.install_log.exists()
    assert env.check_log.exists()
    # The checkout was not pulled.
    assert env.head() == before
    assert "Skipping Git pull" in proc.stdout


def test_no_pull_on_dirty_worktree_still_installs(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    env.make_dirty("modified")
    proc = env.run_update("--no-pull")
    assert proc.returncode == 0, proc.stderr
    env.assert_not_ran(env.pull_log)
    assert env.install_log.exists()
    assert env.check_log.exists()
    assert "Skipping Git pull" in proc.stdout


def test_default_ff_pull_pulls_and_reinstalls(tmp_path: Path) -> None:
    """The happy path: a fast-forward-only pull succeeds, then the checkout is
    reinstalled and the post-install check runs."""
    env = _prepare(tmp_path)

    # Put one commit ahead on the remote so the pull has something to
    # fast-forward to.
    wt = env.work / "wt"
    _git(env.repo, "worktree", "add", "--detach", str(wt), "origin/master")
    (wt / "marker.txt").write_text("pulled\n", encoding="utf-8")
    _git(wt, "add", "marker.txt")
    _git(wt, "commit", "-m", "pulled commit")
    _git(wt, "push", "origin", "HEAD:refs/heads/master")
    remote_head = _git(wt, "rev-parse", "HEAD").stdout.strip()
    _git(env.repo, "worktree", "remove", "--force", str(wt))

    before = env.head()
    proc = env.run_update()
    assert proc.returncode == 0, proc.stderr
    assert env.pull_log.exists(), "expected a fast-forward pull to happen"
    assert env.install_log.exists()
    assert env.check_log.exists()
    # The working tree actually fast-forwarded to the remote commit.
    assert env.head() == remote_head
    assert env.head() != before
    assert (env.repo / "marker.txt").exists()
    assert "Caracal RAG update/reinstall completed." in proc.stdout
    assert "No MCP process was restarted automatically." in proc.stdout
    assert "mcp.transport=stdio" in proc.stdout
    assert "mcp.transport=http" in proc.stdout
    assert "never guesses process ownership or kills processes by name" in proc.stdout


def test_installer_failure_propagates_no_post_install_check(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull", extra_env={"INSTALL_FAIL": "1"})
    assert proc.returncode == 5, f"expected installer exit 5, got {proc.returncode}"
    assert env.install_log.exists()
    env.assert_not_ran(env.check_log)
    assert "active venv was not changed" in proc.stderr
    assert not (env.caracal_home / "venv").is_symlink()
    _assert_no_candidate_generations(env)
    _assert_existing_active_venv_works(env)


def test_post_install_check_failure_propagates_nonzero(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull", extra_env={"CHECK_FAIL": "1"})
    assert proc.returncode == 9, f"expected check exit 9, got {proc.returncode}"
    assert env.install_log.exists()
    assert env.check_log.exists()
    assert "previous active venv remains unchanged" in proc.stderr
    assert not (env.caracal_home / "venv").is_symlink()
    _assert_no_candidate_generations(env)
    _assert_existing_active_venv_works(env)


def test_skip_check_skips_only_post_install_check(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull", "--skip-check", extra_env={"CHECK_FAIL": "1"})
    assert proc.returncode == 0, proc.stderr
    # The installer still ran, but the (failing) check did not.
    assert env.install_log.exists()
    env.assert_not_ran(env.check_log)
    assert "Skipping post-install check" in proc.stdout


def test_caracal_home_passed_through_consistently(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull")
    assert proc.returncode == 0, proc.stderr
    install_line = env.install_log.read_text(encoding="utf-8").strip()
    check_line = env.check_log.read_text(encoding="utf-8").strip()
    expected = f"CARACAL_HOME={env.caracal_home}"
    assert expected in install_line, f"installer saw {install_line!r}"
    assert "CARACAL_INSTALL_WRAPPER=false" in install_line
    assert expected in check_line, f"check saw {check_line!r}"
    # The post-install check ran against the CLI installed under that home.
    assert env.fake_cli.is_file()


def test_unknown_argument_fails_exit_2_with_usage(tmp_path: Path, capfd) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--definitely-not-a-real-flag")
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}"
    combined = proc.stdout + proc.stderr
    assert "ERROR: unknown option" in combined
    assert "Usage:" in combined
    # Argument parsing failed before any Git or install work.
    env.assert_not_ran(env.pull_log, env.install_log, env.check_log)


def test_real_home_is_never_touched(tmp_path: Path) -> None:
    """Every test runs with HOME redirected to a temp dir, so the real user's
    ``~/.caracal`` must never gain a test artifact."""
    env = _prepare(tmp_path)
    env.run_update("--no-pull")
    if REAL_CARACAL_HOME.is_dir():
        test_artifacts = list(REAL_CARACAL_HOME.rglob("pull.log")) + list(
            REAL_CARACAL_HOME.rglob("install.log")
        ) + list(REAL_CARACAL_HOME.rglob("check.log"))
        assert not test_artifacts, f"tests leaked into real home: {test_artifacts}"
    assert str(env.work) == str(tmp_path), "sandbox must stay under tmp_path"



def test_dependency_install_failure_cleans_candidate_and_keeps_active_venv(
    tmp_path: Path,
) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update(
        "--no-pull",
        extra_env={"INSTALL_FAIL_STAGE": "dependencies"},
    )

    assert proc.returncode == 5
    assert "candidate installation failed" in proc.stderr
    assert not (env.caracal_home / "venv").is_symlink()
    assert env.install_log.exists()
    env.assert_not_ran(env.check_log)
    _assert_no_candidate_generations(env)
    _assert_existing_active_venv_works(env)


def test_package_install_failure_after_partial_candidate_keeps_active_venv(
    tmp_path: Path,
) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update(
        "--no-pull",
        extra_env={"INSTALL_FAIL_STAGE": "package"},
    )

    assert proc.returncode == 6
    assert "candidate installation failed" in proc.stderr
    assert not (env.caracal_home / "venv").is_symlink()
    env.assert_not_ran(env.check_log)
    _assert_no_candidate_generations(env)
    _assert_existing_active_venv_works(env)


def test_missing_candidate_entry_point_refuses_promotion(tmp_path: Path) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull", extra_env={"MISSING_ENTRYPOINT": "1"})

    assert proc.returncode != 0
    assert "did not create" in proc.stderr
    assert not (env.caracal_home / "venv").is_symlink()
    env.assert_not_ran(env.check_log)
    _assert_no_candidate_generations(env)
    _assert_existing_active_venv_works(env)


def test_successful_update_promotes_candidate_and_retains_previous_venv(
    tmp_path: Path,
) -> None:
    env = _prepare(tmp_path)
    previous_cli = (env.caracal_home / "venv" / "bin" / "caracal-rag").read_text(
        encoding="utf-8"
    )

    proc = env.run_update("--no-pull")

    active = env.caracal_home / "venv"
    generations = env.caracal_home / "venvs"
    assert proc.returncode == 0, proc.stderr
    assert CANDIDATE_REMAINS_ACTIVE_WARNING not in proc.stderr
    assert active.is_symlink()
    candidate = active.resolve()
    assert candidate.parent == generations.resolve()
    assert candidate.name.startswith("caracal-venv-")
    assert (candidate / "bin" / "python").is_file()
    assert (candidate / "bin" / "caracal-rag").is_file()
    legacy = list(generations.glob("caracal-venv-legacy-*"))
    assert len(legacy) == 1
    assert (legacy[0] / "bin" / "caracal-rag").read_text(encoding="utf-8") == previous_cli
    assert "Previous venv retained for rollback" in proc.stdout
    assert env.check_log.exists()
    first_candidate = candidate
    # Stable wrapper must dispatch through the newly active generation.
    active_check = _run(
        [str(env.fake_cli), "check"], cwd=env.repo, env=dict(env.base_env)
    )
    assert active_check.returncode == 0, active_check.stderr
    assert "check ok" in active_check.stdout

    # A second update must work when venv is already a generation symlink,
    # retaining the previous generation instead of attempting to move it.
    second = env.run_update("--no-pull")
    assert second.returncode == 0, second.stderr
    assert CANDIDATE_REMAINS_ACTIVE_WARNING not in second.stderr
    second_candidate = active.resolve()
    assert second_candidate != first_candidate
    assert second_candidate.is_dir()
    assert first_candidate.is_dir()
    assert (first_candidate / "bin" / "caracal-rag").is_file()
    assert "Previous venv retained for rollback" in second.stdout


def test_skip_check_still_validates_entry_point_and_promotes_candidate(
    tmp_path: Path,
) -> None:
    env = _prepare(tmp_path)
    proc = env.run_update("--no-pull", "--skip-check")

    active = env.caracal_home / "venv"
    assert proc.returncode == 0, proc.stderr
    assert CANDIDATE_REMAINS_ACTIVE_WARNING not in proc.stderr
    assert active.is_symlink()
    assert (active.resolve() / "bin" / "caracal-rag").is_file()
    env.assert_not_ran(env.check_log)
    assert "Skipping candidate connectivity check" in proc.stdout



def test_candidate_check_failure_keeps_existing_managed_generation_active(
    tmp_path: Path,
) -> None:
    env = _prepare(tmp_path)
    first = env.run_update("--no-pull")
    assert first.returncode == 0, first.stderr

    active = env.caracal_home / "venv"
    original_target = active.resolve()
    original_generations = set((env.caracal_home / "venvs").glob("caracal-venv-*"))
    assert active.is_symlink()
    assert original_target.is_dir()

    failed = env.run_update("--no-pull", extra_env={"CHECK_FAIL": "1"})

    assert failed.returncode == 9
    assert active.is_symlink()
    assert active.resolve() == original_target
    assert original_target.is_dir()
    assert set((env.caracal_home / "venvs").glob("caracal-venv-*")) == original_generations
    _assert_existing_active_venv_works(env)
