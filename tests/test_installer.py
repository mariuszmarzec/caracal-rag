from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_python_script() -> str:
    venv_python = r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
    printf '%s\n' "${FAKE_PYTHON_VERSION:-3.12}"
    exit 0
fi
printf '%s\n' "$*" >> "${CARACAL_HOME}/venv-install.log"
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    if [[ " $* " == *" -e "* ]]; then
        echo "editable installs are forbidden in this test" >&2
        exit 22
    fi
fi
exit 0
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "venv" ]]; then
    venv="$3"
    mkdir -p "$venv/bin"
    cat >"$venv/bin/python" <<'FAKE_VENV_PYTHON'
{venv_python.rstrip()}
FAKE_VENV_PYTHON
    chmod +x "$venv/bin/python"
    cat >"$venv/bin/caracal-rag" <<'FAKE_CLI'
#!/usr/bin/env bash
printf 'fake installed CLI invoked\\n'
FAKE_CLI
    chmod +x "$venv/bin/caracal-rag"
    exit 0
fi
echo "unexpected base Python invocation: $*" >&2
exit 23
"""


def _install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "install.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_installer_uses_private_venv_and_preserves_existing_state(tmp_path: Path) -> None:
    home = tmp_path / "caracal-home"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(_fake_python_script(), encoding="utf-8")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["CARACAL_HOME"] = str(home)
    env["CARACAL_PYTHON"] = str(fake_python)

    first = _install(env)
    assert first.returncode == 0, first.stderr
    venv_python = home / "venv" / "bin" / "python"
    installed_cli = home / "bin" / "caracal-rag"
    assert venv_python.is_file()
    assert installed_cli.is_file()
    assert (home / "config" / "sources.yaml").is_file()
    secrets_path = home / "config" / "secrets.yaml"
    assert secrets_path.stat().st_mode & 0o777 == 0o600

    install_args = (home / "venv-install.log").read_text(encoding="utf-8")
    assert "pip install --constraint" in install_args
    assert f"--constraint {REPO_ROOT / 'constraints-py312.lock'}" in install_args
    assert "--upgrade --force-reinstall" in install_args
    assert " -e " not in f" {install_args} "

    cli = subprocess.run(
        [str(installed_cli)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert "fake installed CLI invoked" in cli.stdout

    config_path = home / "config" / "sources.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\ncustom_install_test: preserve-me\n",
        encoding="utf-8",
    )
    secrets_path.write_text("embedding:\n  api_key: preserve-me\n", encoding="utf-8")
    secrets_path.chmod(0o644)
    manifest_path = home / "index" / "manifest.json"
    manifest_path.write_text('{"embedding_model":"bge-m3"}\n', encoding="utf-8")

    second = _install(env)
    assert second.returncode == 0, second.stderr
    assert "custom_install_test: preserve-me" in config_path.read_text(encoding="utf-8")
    assert "api_key: preserve-me" in secrets_path.read_text(encoding="utf-8")
    assert secrets_path.stat().st_mode & 0o777 == 0o600
    assert manifest_path.read_text(encoding="utf-8") == (
        '{"embedding_model":"bge-m3"}\n'
    )
    assert (home / "venv-install.log").read_text(encoding="utf-8").count(
        "--force-reinstall"
    ) == 2



def test_installer_refuses_python_without_a_lock_before_pip_install(tmp_path: Path) -> None:
    home = tmp_path / "caracal-home"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(_fake_python_script(), encoding="utf-8")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["CARACAL_HOME"] = str(home)
    env["CARACAL_PYTHON"] = str(fake_python)
    env["FAKE_PYTHON_VERSION"] = "3.15"

    result = _install(env)

    assert result.returncode != 0
    assert "no locked dependency set is available for Python 3.15" in result.stderr
    assert not (home / "venv-install.log").exists()



def test_installer_can_build_candidate_venv_without_replacing_active_wrapper(
    tmp_path: Path,
) -> None:
    """Transactional update installation uses a separate venv and leaves wrapper untouched."""
    home = tmp_path / "caracal-home"
    candidate = home / "venvs" / "candidate-one"
    wrapper = home / "bin" / "caracal-rag"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env bash\necho old-wrapper\n", encoding="utf-8")
    wrapper.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(_fake_python_script(), encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["CARACAL_HOME"] = str(home)
    env["CARACAL_PYTHON"] = str(fake_python)
    env["CARACAL_VENV"] = str(candidate)
    env["CARACAL_INSTALL_WRAPPER"] = "false"

    result = _install(env)

    assert result.returncode == 0, result.stderr
    assert (candidate / "bin" / "python").is_file()
    assert (candidate / "bin" / "caracal-rag").is_file()
    assert not (home / "venv").exists()
    assert wrapper.read_text(encoding="utf-8") == (
        "#!/usr/bin/env bash\necho old-wrapper\n"
    )
    install_args = (home / "venv-install.log").read_text(encoding="utf-8")
    assert f"--constraint {REPO_ROOT / 'constraints-py312.lock'}" in install_args
    assert "Leaving the active CLI wrapper unchanged" in result.stdout
