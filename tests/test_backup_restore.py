"""Safety tests for local Caracal state backup/restore scripts.

All files and shell subprocesses run under pytest temporary directories. These
tests never read or modify the real user's ~/.caracal and never contact Chroma.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = REPO_ROOT / "backup.sh"
RESTORE_SCRIPT = REPO_ROOT / "restore.sh"

API_KEY = "backup-test-api-key-must-not-print"
ACCESS_TOKEN = "backup-test-access-token-must-not-print"


def _run(
    script: Path,
    *args: str,
    home: Path,
    caracal_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("CARACAL_HOME", None)
    env.pop("CARACAL_BACKUP_ROOT", None)
    if caracal_home is not None:
        env["CARACAL_HOME"] = str(caracal_home)
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _make_state(root: Path, *, manifest: bool = True) -> Path:
    state = root / ".caracal"
    config = state / "config"
    index = state / "index"
    config.mkdir(parents=True)
    index.mkdir(parents=True)
    (config / "sources.yaml").write_text(
        "embedding:\n  model: bge-m3\n"
        "  api_base: http://localhost:3001/v1\n"
        "chroma:\n  host: localhost\n  port: 3400\n  collection: caracal-test\n"
        "access_token:\n  enabled: true\n",
        encoding="utf-8",
    )
    (config / "secrets.yaml").write_text(
        f"embedding:\n  api_key: {API_KEY}\n"
        f"access_token:\n  token: {ACCESS_TOKEN}\n",
        encoding="utf-8",
    )
    (config / "secrets.yaml").chmod(0o644)
    if manifest:
        (index / "manifest.json").write_text(
            '{"collection":"caracal-test","embedding_model":"bge-m3",'
            '"embedding_api_base":"http://localhost:3001/v1",'
            '"embedding_dim":1024,"document_count":3}\n',
            encoding="utf-8",
        )
    return state


def _backup(
    source_home: Path, backup_path: Path, *, home: Path
) -> subprocess.CompletedProcess[str]:
    return _run(
        BACKUP_SCRIPT,
        "--home",
        str(source_home / ".caracal"),
        "--output",
        str(backup_path),
        home=home,
    )


def _file_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_backup_contains_only_local_state_and_restricts_permissions(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    state = _make_state(source_home)
    destination = tmp_path / "backups" / "snapshot"
    proc = _backup(source_home, destination, home=tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert destination.is_dir()
    assert (destination / "config" / "sources.yaml").read_bytes() == (
        state / "config" / "sources.yaml"
    ).read_bytes()
    assert (destination / "config" / "secrets.yaml").read_bytes() == (
        state / "config" / "secrets.yaml"
    ).read_bytes()
    assert (destination / "index" / "manifest.json").read_bytes() == (
        state / "index" / "manifest.json"
    ).read_bytes()
    assert "chroma_vectors=included:NO" in (
        destination / "BACKUP-INFO.txt"
    ).read_text(encoding="utf-8")

    assert _file_mode(destination) == 0o700
    assert _file_mode(destination / "config") == 0o700
    assert _file_mode(destination / "index") == 0o700
    assert _file_mode(destination / "config" / "sources.yaml") == 0o600
    assert _file_mode(destination / "config" / "secrets.yaml") == 0o600
    assert _file_mode(destination / "index" / "manifest.json") == 0o600
    assert API_KEY not in proc.stdout + proc.stderr
    assert ACCESS_TOKEN not in proc.stdout + proc.stderr


def test_backup_defaults_outside_repository_and_records_missing_manifest(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home, manifest=False)
    proc = _run(
        BACKUP_SCRIPT,
        "--home",
        str(source_home / ".caracal"),
        home=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    backup_root = tmp_path / "caracal-backups"
    backups = list(backup_root.glob("caracal-*"))
    assert len(backups) == 1
    assert backups[0].parent == backup_root
    assert "manifest_json=absent" in (
        backups[0] / "BACKUP-INFO.txt"
    ).read_text(encoding="utf-8")
    assert not (backups[0] / "index" / "manifest.json").exists()


def test_backup_refuses_destination_inside_repository(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    forbidden = REPO_ROOT / ".test-backup-must-not-be-created"
    assert not forbidden.exists()
    proc = _backup(source_home, forbidden, home=tmp_path)
    assert proc.returncode != 0
    assert "inside the source repository" in proc.stderr
    assert not forbidden.exists()


def test_backup_refuses_symlinked_config_directory(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    external = tmp_path / "external-config"
    external.mkdir()
    (external / "sources.yaml").write_text("sources: []\n", encoding="utf-8")
    (external / "secrets.yaml").write_text("embedding: {}\n", encoding="utf-8")
    state = source_home / ".caracal"
    state.mkdir(parents=True)
    (state / "config").symlink_to(external, target_is_directory=True)
    (state / "index").mkdir()
    destination = tmp_path / "backup"
    proc = _backup(source_home, destination, home=tmp_path)
    assert proc.returncode != 0
    assert "symlink local-state directory" in proc.stderr
    assert not destination.exists()


def test_restore_into_fresh_home_does_not_require_force_and_restores_modes(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr

    restore_home = tmp_path / "restore"
    destination = restore_home / ".caracal"
    restored = _run(
        RESTORE_SCRIPT,
        str(backup),
        "--home",
        str(destination),
        home=restore_home,
    )

    assert restored.returncode == 0, restored.stderr
    assert (destination / "config" / "sources.yaml").read_bytes() == (
        backup / "config" / "sources.yaml"
    ).read_bytes()
    assert (destination / "config" / "secrets.yaml").read_bytes() == (
        backup / "config" / "secrets.yaml"
    ).read_bytes()
    assert (destination / "index" / "manifest.json").read_bytes() == (
        backup / "index" / "manifest.json"
    ).read_bytes()
    assert _file_mode(destination / "config" / "secrets.yaml") == 0o600
    assert _file_mode(destination / "config" / "sources.yaml") == 0o600
    assert _file_mode(destination / "index" / "manifest.json") == 0o600
    assert "Remote Chroma vectors were not restored" in restored.stdout
    assert API_KEY not in restored.stdout + restored.stderr
    assert ACCESS_TOKEN not in restored.stdout + restored.stderr


def test_restore_refuses_to_replace_existing_state_without_force(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr

    target_home = tmp_path / "target"
    target_state = _make_state(target_home)
    previous = {
        path.relative_to(target_state): path.read_bytes()
        for path in target_state.rglob("*")
        if path.is_file()
    }
    proc = _run(
        RESTORE_SCRIPT,
        str(backup),
        "--home",
        str(target_state),
        home=tmp_path,
    )

    assert proc.returncode != 0
    assert "No files were restored" in proc.stderr
    for relative, data in previous.items():
        assert (target_state / relative).read_bytes() == data


def test_restore_force_preserves_old_state_before_replacing(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr

    target_home = tmp_path / "target"
    target_state = _make_state(target_home)
    target_sources = target_state / "config" / "sources.yaml"
    target_sources.write_text("custom: preserve-before-restore\n", encoding="utf-8")
    old_secrets = "embedding:\n  api_key: old-local-key\n"
    (target_state / "config" / "secrets.yaml").write_text(old_secrets, encoding="utf-8")

    proc = _run(
        RESTORE_SCRIPT,
        str(backup),
        "--home",
        str(target_state),
        "--force",
        home=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    assert target_sources.read_bytes() == (
        backup / "config" / "sources.yaml"
    ).read_bytes()
    assert (target_state / "config" / "secrets.yaml").read_bytes() == (
        backup / "config" / "secrets.yaml"
    ).read_bytes()
    snapshots = list((tmp_path / "caracal-backups").glob("pre-restore-*"))
    assert len(snapshots) == 1
    safety = snapshots[0]
    assert (safety / "config" / "sources.yaml").read_text(encoding="utf-8") == (
        "custom: preserve-before-restore\n"
    )
    assert (safety / "config" / "secrets.yaml").read_text(encoding="utf-8") == old_secrets
    assert _file_mode(safety) == 0o700
    assert _file_mode(safety / "config" / "secrets.yaml") == 0o600
    assert API_KEY not in proc.stdout + proc.stderr
    assert ACCESS_TOKEN not in proc.stdout + proc.stderr


def test_restore_backup_without_manifest_removes_existing_manifest_only_with_force(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home, manifest=False)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr

    target_home = tmp_path / "target"
    target_state = _make_state(target_home, manifest=True)
    failed = _run(
        RESTORE_SCRIPT, str(backup), "--home", str(target_state), home=tmp_path
    )
    assert failed.returncode != 0
    assert (target_state / "index" / "manifest.json").exists()

    succeeded = _run(
        RESTORE_SCRIPT,
        str(backup),
        "--home",
        str(target_state),
        "--force",
        home=tmp_path,
    )
    assert succeeded.returncode == 0, succeeded.stderr
    assert not (target_state / "index" / "manifest.json").exists()
    safety = list((tmp_path / "caracal-backups").glob("pre-restore-*"))
    assert len(safety) == 1
    assert (safety[0] / "index" / "manifest.json").exists()


def test_restore_refuses_symlinked_target_config_directory(tmp_path: Path) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr

    target_home = tmp_path / "target"
    target_state = target_home / ".caracal"
    target_state.mkdir(parents=True)
    external = tmp_path / "external-config"
    external.mkdir()
    (external / "do-not-write.txt").write_text("unchanged\n", encoding="utf-8")
    (target_state / "config").symlink_to(external, target_is_directory=True)

    proc = _run(
        RESTORE_SCRIPT, str(backup), "--home", str(target_state), home=tmp_path
    )
    assert proc.returncode != 0
    assert "symlink local-state directory" in proc.stderr
    assert not (external / "sources.yaml").exists()
    assert (external / "do-not-write.txt").read_text(encoding="utf-8") == "unchanged\n"


def test_restore_rejects_backup_missing_declared_manifest_without_writing(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source"
    _make_state(source_home)
    backup = tmp_path / "backup"
    created = _backup(source_home, backup, home=tmp_path)
    assert created.returncode == 0, created.stderr
    (backup / "index" / "manifest.json").unlink()

    target_home = tmp_path / "fresh-target"
    target_state = target_home / ".caracal"
    proc = _run(
        RESTORE_SCRIPT,
        str(backup),
        "--home",
        str(target_state),
        home=target_home,
    )

    assert proc.returncode != 0
    assert "manifest is included, but the file is missing" in proc.stderr
    assert not (target_state / "config" / "sources.yaml").exists()
