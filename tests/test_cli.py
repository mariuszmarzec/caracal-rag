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

from caracal_rag.cli import run_check  # noqa: E402


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


class _FakeChromaClient:
    def __init__(self, heartbeat: int = 1) -> None:
        self._heartbeat = heartbeat

    def heartbeat(self) -> int:
        if self._heartbeat == 0:
            raise RuntimeError("Chroma heartbeat failed")
        return self._heartbeat


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
