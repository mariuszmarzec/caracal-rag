"""Tests for the ``caracal_rag check`` command.

The ``check`` command must validate the configuration and actually connect to
the configured LiteLLM and Chroma endpoints. These tests stub both services so
they run without network access, exercising the success and failure paths.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

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

from caracal_rag.cli import build_parser, run_check, run_index, run_mcp, run_setup  # noqa: E402
from caracal_rag.config import AppConfig, ConfigError  # noqa: E402

#: Every protected interface path requires a valid token. These tests stub the
#: embedding/Chroma services, so they supply a token through the environment
#: variable the runtime reads -- never through a command-line argument or a
#: checked-in secret.
_TEST_TOKEN = "test-access-token-1234"


@pytest.fixture(autouse=True)
def _access_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARACAL_ACCESS_TOKEN", _TEST_TOKEN)


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    data = {
        "embedding": {
            "model": "bge-m3",
            "api_base": "http://localhost:3001/v1",
            "api_key": "test-key",
        },
        "chroma": {"host": "localhost", "port": 8000, "collection": "test"},
        "access_token": {"enabled": True},
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
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
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
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
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
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
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
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
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

    def failing_probe(*args, **kwargs):
        raise RuntimeError("LiteLLM unreachable")

    monkeypatch.setattr("caracal_rag.cli.probe_embedding_dim", failing_probe)

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


def test_mcp_parser_defaults_and_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(["mcp"])
    assert args.transport is None
    assert args.host is None
    assert args.port is None
    assert args.path is None
    assert args.config is None

    args = parser.parse_args(["mcp", "--transport", "http", "--host", "0.0.0.0",
                              "--port", "9000", "--path", "/caracal", "--config", "x.yaml"])
    assert args.transport == "http"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.path == "/caracal"
    assert args.config == "x.yaml"


def test_mcp_parser_rejects_unknown_transport() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["mcp", "--transport", "ws"])
    assert excinfo.value.code == 2


def test_run_mcp_dispatches_http(monkeypatch: pytest.MonkeyPatch, example_config: Path) -> None:
    """``run_mcp`` must dispatch to the http or stdio transport based on config."""
    calls: dict[str, Any] = {}

    class _FakeServer:
        def __init__(self, config: AppConfig) -> None:
            calls["config"] = config

        async def run_http_async(self) -> None:
            calls["ran"] = "http"

        async def run_stdio_async(self) -> None:
            calls["ran"] = "stdio"

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _FakeChromaClient(heartbeat=1),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
    )
    monkeypatch.setattr("caracal_rag.mcp.CaracalMcpServer", _FakeServer)

    class _FakeAsyncio:
        @staticmethod
        def run(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()

    monkeypatch.setattr("caracal_rag.cli.asyncio", _FakeAsyncio)

    run_mcp(transport_override="http", config_path=str(example_config))
    assert calls["ran"] == "http"
    assert calls["config"].mcp.transport == "http"

    calls.clear()
    run_mcp(transport_override="stdio", config_path=str(example_config))
    assert calls["ran"] == "stdio"


def test_run_mcp_uses_env_transport(monkeypatch: pytest.MonkeyPatch, example_config: Path) -> None:
    """The transport falls back to the env-aware config when no CLI override is given."""
    class _FakeServer:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        async def run_http_async(self) -> None:
            pass

        async def run_stdio_async(self) -> None:
            raise AssertionError("stdio must not be dispatched when env says http")

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _FakeChromaClient(heartbeat=1),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 3,
    )
    monkeypatch.setattr("caracal_rag.mcp.CaracalMcpServer", _FakeServer)

    class _FakeAsyncio:
        @staticmethod
        def run(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()

    monkeypatch.setattr("caracal_rag.cli.asyncio", _FakeAsyncio)
    monkeypatch.setenv("CARACAL_MCP_TRANSPORT", "http")

    run_mcp(config_path=str(example_config))
    # Reaching the end without AssertionError means http was dispatched from
    # the env-aware config, not from a CLI override.


def _write_manifest(home: Path, *, model: str, dim: int | None) -> Path:
    index_dir = home / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    payload = {
        "collection": "test",
        "embedding_model": model,
        "embedding_api_base": "http://localhost:3001/v1",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "document_count": 1,
    }
    if dim is not None:
        payload["embedding_dim"] = dim
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return manifest_path


def test_check_refuses_dimension_drift_against_manifest(
    monkeypatch: pytest.MonkeyPatch, example_config: Path, tmp_path: Path
) -> None:
    """A manifest recording a different dimension must fail ``check``.

    The configured model was probed at 512 dims but the manifest records the
    index was built with 1024-dim vectors: the stored vectors live in a
    different embedding space and must be refused with rebuild guidance, even
    when the collection model tag agrees with the configured model.
    """
    home = tmp_path / "caracal-home"
    home.mkdir()
    _write_manifest(home, model="bge-m3", dim=1024)
    monkeypatch.setenv("CARACAL_HOME", str(home))

    class _SameModelCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "bge-m3"}],
            }

        def query(self, **kwargs):
            return {"ids": [["a"]], "metadatas": [[{"embedding_model": "bge-m3"}]]}

    class _ChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _SameModelCollection:
            return _SameModelCollection()

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _ChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 512,
    )

    manifest_path = home / "index" / "manifest.json"
    manifest_before = manifest_path.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0

    # A failed check must not rewrite the manifest.
    assert manifest_path.read_text(encoding="utf-8") == manifest_before


def test_check_refuses_dimension_drift_against_collection(
    monkeypatch: pytest.MonkeyPatch, example_config: Path, tmp_path: Path
) -> None:
    """A live collection with a different stored dimension must fail ``check``.

    The manifest may be absent or agree on model, but the *collection* itself
    is inspected: committed vectors tagged with a different ``embedding_dim``
    must be refused with rebuild guidance.
    """
    home = tmp_path / "caracal-home"
    home.mkdir()
    monkeypatch.setenv("CARACAL_HOME", str(home))

    class _DimDriftCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "bge-m3", "embedding_dim": 1024}],
                "embeddings": [[0.1] * 1024],
            }

        def query(self, **kwargs):
            return {
                "ids": [["a"]],
                "metadatas": [[{"embedding_model": "bge-m3", "embedding_dim": 1024}]],
            }

    class _ChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _DimDriftCollection:
            return _DimDriftCollection()

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _ChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 512,
    )

    with pytest.raises(SystemExit) as excinfo:
        run_check(config_path=str(example_config))
    assert excinfo.value.code != 0


def test_index_refuses_dimension_drift_before_writing(
    monkeypatch: pytest.MonkeyPatch, example_config: Path, tmp_path: Path
) -> None:
    """A failed indexing run (dimension drift) must not update the manifest.

    The manifest records the index was built with 1024-dim vectors while the
    live probe returns 512 dims. ``index`` must refuse to run and must leave
    the manifest untouched.
    """
    home = tmp_path / "caracal-home"
    home.mkdir()
    manifest_path = _write_manifest(home, model="bge-m3", dim=1024)
    monkeypatch.setenv("CARACAL_HOME", str(home))

    class _DriftCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "bge-m3", "embedding_dim": 1024}],
                "embeddings": [[0.1] * 1024],
            }

    class _ChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _DriftCollection:
            return _DriftCollection()

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _ChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 512,
    )

    with pytest.raises(ConfigError, match="dimensional"):
        run_index(source=None, config_path=str(example_config))

    manifest_before = manifest_path.read_text(encoding="utf-8")
    assert '"embedding_dim": 1024' in manifest_before


def test_index_success_writes_dimension_to_manifest(
    monkeypatch: pytest.MonkeyPatch, example_config: Path, tmp_path: Path
) -> None:
    """A successful indexing run persists the probed dimension in the manifest.

    The manifest is only written after the indexer completes successfully, so
    the dimension recorded is the one that actually produced the stored
    vectors.
    """
    home = tmp_path / "caracal-home"
    home.mkdir()
    monkeypatch.setenv("CARACAL_HOME", str(home))

    class _FakeCollection:
        def __init__(self) -> None:
            self._rows: dict[str, dict] = {}

        def get(self, where=None, include=None):
            result: dict = {"ids": list(self._rows.keys())}
            if include and "metadatas" in include:
                result["metadatas"] = [
                    record.get("metadata", record) for record in self._rows.values()
                ]
            if include and "documents" in include:
                result["documents"] = [
                    record.get("text", "") for record in self._rows.values()
                ]
            if include and "embeddings" in include:
                result["embeddings"] = [
                    record.get("embedding") for record in self._rows.values()
                ]
            return result

        def upsert(self, ids, documents, embeddings, metadatas):
            for cid, text, emb, meta in zip(ids, documents, embeddings, metadatas):
                self._rows[cid] = {"text": text, "embedding": emb, "metadata": meta, **meta}

        def delete(self, ids):
            for cid in ids:
                self._rows.pop(cid, None)

        def query(self, **kwargs):
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    collection = _FakeCollection()

    class _ChromaClient:
        def heartbeat(self) -> int:
            return 1

        def get_or_create_collection(self, name: str) -> _FakeCollection:
            return collection

    monkeypatch.setattr(
        "caracal_rag.cli._chroma_client",
        lambda *args, **kwargs: _ChromaClient(),
    )
    monkeypatch.setattr(
        "caracal_rag.cli.probe_embedding_dim",
        lambda api_base, model, api_key=None: 7,
    )

    # Stub out the indexer so run_index runs end-to-end without a real source.
    class _FakeIndexer:
        def __init__(self, config, source_filter=None, embedding_dim=None):
            self.config = config
            self.collection = collection
            self.embedding_dim = embedding_dim

        def run(self):
            collection.upsert(
                ids=["chunk-1"],
                documents=["hello"],
                embeddings=[[0.1] * 7],
                metadatas=[
                    {
                        "url": "https://example.com/a.md",
                        "embedding_model": "bge-m3",
                        "embedding_dim": self.embedding_dim,
                        "caracal_staging": False,
                    }
                ],
            )

    monkeypatch.setattr("caracal_rag.cli.Indexer", _FakeIndexer)

    run_index(source=None, config_path=str(example_config))

    manifest_path = home / "index" / "manifest.json"
    assert manifest_path.is_file(), "manifest was not written after successful indexing"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["embedding_dim"] == 7
    assert data["embedding_model"] == "bge-m3"
    assert data["document_count"] == 1



def test_incomplete_index_run_does_not_update_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial source inventory must not make CLI write a successful-run manifest."""
    import caracal_rag.cli as cli_module

    class _Config:
        embedding = object()

    class _FakeIndexer:
        def __init__(self, config, source_filter=None, embedding_dim=None):
            pass

        def run(self):
            return False

    manifest_writes: list[bool] = []
    monkeypatch.setattr(cli_module, "load_active_config", lambda config_path=None: _Config())
    monkeypatch.setattr(cli_module, "_ensure_api_key", lambda config: None)
    monkeypatch.setattr(cli_module, "_probe_dim", lambda config: 2)
    monkeypatch.setattr(cli_module, "_assert_index_safe", lambda config, observed_dim=None: None)
    monkeypatch.setattr(cli_module, "Indexer", _FakeIndexer)
    monkeypatch.setattr(
        cli_module,
        "_write_manifest",
        lambda *args, **kwargs: manifest_writes.append(True),
    )

    with pytest.raises(RuntimeError, match="source inventory was incomplete"):
        run_index(source=None, config_path="unused-test-config.yaml")

    assert manifest_writes == []
