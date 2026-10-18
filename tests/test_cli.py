"""Tests for the ``caracal_rag check`` command.

The ``check`` command must validate the configuration and actually connect to
the configured LiteLLM and Chroma endpoints. These tests stub both services so
they run without network access, exercising the success and failure paths.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

# Stub out heavy optional dependencies so importing caracal_rag.cli works in CI.
chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

litellm_stub = types.ModuleType("litellm")


class _FakeLiteLLM:
    api_base = None
    api_key = None

    @staticmethod
    def embedding(*args, **kwargs) -> dict:
        return {"data": [{"embedding": [0.1, 0.2]}]}


litellm_stub.embedding = _FakeLiteLLM.embedding
litellm_stub.api_base = None
litellm_stub.api_key = None
sys.modules["litellm"] = litellm_stub

from caracal_rag.cli import build_parser, run_check, run_setup  # noqa: E402
from caracal_rag.config import ConfigError  # noqa: E402


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    data = {
        "embedding": {
            "model": "bge-m3",
            "api_base": "http://localhost:3001/v1",
            "api_key": "test-key",
        },
        "chroma": {"host": "localhost", "port": 8000, "collection": "test"},
        "sources": [
            {
                "name": "api-docs",
                "type": "markdown",
                "urls": ["https://example.com/login.md"],
            }
        ],
    }
    config_path = tmp_path / "sources.example.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


class _EmptyCollection:
    def get(self, include=None):
        return {"ids": [], "metadatas": []}

    def query(self, **kwargs):
        return {"ids": [], "metadatas": []}


class _FakeChromaClient:
    def __init__(self, heartbeat: int = 1) -> None:
        self._heartbeat = heartbeat

    def heartbeat(self) -> int:
        if self._heartbeat == 0:
            raise RuntimeError("Chroma heartbeat failed")
        return self._heartbeat

    def get_or_create_collection(self, name: str) -> _EmptyCollection:
        return _EmptyCollection()


def test_check_validates_connectivity(
    monkeypatch: pytest.MonkeyPatch, example_config: Path, capsys: pytest.CaptureFixture
) -> None:
    """A healthy configuration prints connectivity info and exits cleanly."""
    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _FakeChromaClient(heartbeat=1),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.embed_texts",
        lambda texts, api_base, api_key=None, model=None: [[0.1, 0.2, 0.3]],
    )

    run_check(config_path=str(example_config))

    out = capsys.readouterr().out
    assert "sources=1" in out
    assert "embedding_api_base=http://localhost:3001/v1" in out
    assert "chroma=localhost:8000" in out
    assert "chroma_heartbeat=1" in out
    assert "embedding_dim=3" in out
    assert "ok" in out


def test_check_refuses_mismatched_live_collection(
    monkeypatch: pytest.MonkeyPatch, example_config: Path
) -> None:
    """``check`` must refuse a live collection holding vectors from another model.

    The manifest may be absent or consistent, but the *collection* itself is
    inspected: vectors tagged with a different family (or none at all) must
    never be reported as healthy, because that would let an incompatible index
    be reused for queries.
    """
    class _MismatchedCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "old-family"}],
            }

        def query(self, **kwargs):
            return {"ids": [["a"]], "metadatas": [[{"embedding_model": "old-family"}]]}

    class _MismatchedChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _MismatchedCollection:
            return _MismatchedCollection()

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _MismatchedChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.embed_texts",
        lambda texts, api_base, api_key=None, model=None: [[0.1, 0.2, 0.3]],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0


def test_check_refuses_legacy_live_collection(
    monkeypatch: pytest.MonkeyPatch, example_config: Path
) -> None:
    """A collection holding untagged vectors must not pass ``check``.

    A missing manifest alone is not proof that the index is compatible: the
    live collection is inspected, and vectors without ``embedding_model``
    metadata are refused with rebuild guidance.
    """
    class _LegacyCollection:
        def get(self, include=None):
            return {"ids": ["a", "b"], "metadatas": [{"embedding_model": None}, {}]}

        def query(self, **kwargs):
            return {"ids": [["a"]], "metadatas": [[{"embedding_model": None}]]}

    class _LegacyChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _LegacyCollection:
            return _LegacyCollection()

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _LegacyChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.embed_texts",
        lambda texts, api_base, api_key=None, model=None: [[0.1, 0.2, 0.3]],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0


def test_check_raises_on_chroma_failure(
    monkeypatch: pytest.MonkeyPatch, example_config: Path
) -> None:
    """A Chroma heartbeat failure must abort the check with a non-zero exit."""
    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _FakeChromaClient(heartbeat=0),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.embed_texts",
        lambda texts, api_base, api_key=None, model=None: [[0.1, 0.2]],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0


def test_check_raises_on_embedding_failure(
    monkeypatch: pytest.MonkeyPatch, example_config: Path
) -> None:
    """A LiteLLM probe failure must abort the check with a non-zero exit."""
    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _FakeChromaClient(heartbeat=1),
    )

    def failing_embed(*args, **kwargs):
        raise RuntimeError("LiteLLM unreachable")

    monkeypatch.setattr("caracal_rag.cli.embed_texts", failing_embed)

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0



def _write_active_config(home: Path, *, model: str = "bge-m3") -> Path:
    config_dir = home / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "sources.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "embedding": {
                    "model": model,
                    "api_base": "http://localhost:3001/v1",
                    "provider_extension": {"keep": "yes"},
                },
                "chroma": {
                    "host": "localhost",
                    "port": 8000,
                    "collection": "custom-collection",
                    "custom_chroma_option": "keep",
                },
                "sources": [
                    {
                        "name": "custom-docs",
                        "type": "markdown",
                        "urls": ["https://example.com/custom.md"],
                        "source_extension": {"preserve": True},
                    }
                ],
                "custom_top_level": {"preserve": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    secrets_path = config_dir / "secrets.yaml"
    secrets_path.write_text(
        "embedding:\n  api_key: old-test-key\ncustom_secret_metadata: keep\n",
        encoding="utf-8",
    )
    secrets_path.chmod(0o600)
    return config_path


def test_setup_preserves_user_config_and_rotates_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "caracal-home"
    config_path = _write_active_config(home)
    monkeypatch.setenv("CARACAL_HOME", str(home))
    monkeypatch.setenv("CARACAL_EMBEDDING_API_KEY", "new-test-key")
    monkeypatch.setattr("caracal_rag.cli.validate_embedding_endpoint", lambda **kwargs: None)

    run_setup(model="bge-m3")

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["custom_top_level"] == {"preserve": True}
    assert saved["embedding"]["provider_extension"] == {"keep": "yes"}
    assert "api_key" not in saved["embedding"]
    assert saved["chroma"]["custom_chroma_option"] == "keep"
    assert saved["sources"][0]["source_extension"] == {"preserve": True}
    assert saved["sources"][0]["urls"] == ["https://example.com/custom.md"]

    secrets_path = home / "config" / "secrets.yaml"
    secrets = yaml.safe_load(secrets_path.read_text(encoding="utf-8"))
    assert secrets["embedding"]["api_key"] == "new-test-key"
    assert secrets["custom_secret_metadata"] == "keep"
    assert secrets_path.stat().st_mode & 0o777 == 0o600


def test_setup_refuses_malformed_config_without_replacing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "caracal-home"
    config_dir = home / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "sources.yaml"
    original = "embedding: [this is not valid YAML\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("CARACAL_HOME", str(home))
    monkeypatch.setenv("CARACAL_EMBEDDING_API_KEY", "test-key")

    with pytest.raises(ConfigError, match="left unchanged"):
        run_setup(model="bge-m3")

    assert config_path.read_text(encoding="utf-8") == original


def test_check_missing_config_reports_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("CARACAL_HOME", str(tmp_path / "missing-home"))
    monkeypatch.delenv("CARACAL_EMBEDDING_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        run_check()

    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "No active configuration found" in err
    assert "UnboundLocalError" not in err


def test_setup_does_not_accept_key_as_command_argument() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["setup", "--api-key", "should-not-be-a-cli-argument"])
    assert excinfo.value.code == 2
