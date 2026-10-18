from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

# Stub out heavy optional dependencies so importing caracal_rag.vectorstore works
# without chromadb being importable in CI.
chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

from caracal_rag.config import AppConfig, ConfigError  # noqa: E402
from caracal_rag.manifest import (  # noqa: E402
    assert_manifest_consistent,
    build_manifest,
    load_manifest,
    write_manifest,
)
from caracal_rag.vectorstore import assert_embedding_model_consistent  # noqa: E402


def _config() -> AppConfig:
    return AppConfig.from_yaml("config/sources.example.yaml")


def test_manifest_round_trip_and_consistency(tmp_path: Path) -> None:
    config = _config()
    manifest_path = tmp_path / "index" / "manifest.json"

    assert load_manifest(manifest_path) is None
    # First-time indexing can establish the manifest when no index is recorded.
    assert_manifest_consistent(manifest_path, config)

    write_manifest(manifest_path, build_manifest(config, document_count=7))
    saved = load_manifest(manifest_path)
    assert saved is not None
    assert saved.embedding_model == config.embedding.model
    assert saved.document_count == 7
    assert_manifest_consistent(manifest_path, config)


def test_manifest_mismatch_refuses_to_reuse_index(tmp_path: Path) -> None:
    config = _config()
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, build_manifest(config, document_count=3))

    config.embedding.model = "different-explicit-family"
    with pytest.raises(ConfigError, match="Mixing embedding spaces"):
        assert_manifest_consistent(manifest_path, config)


def test_corrupt_manifest_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{ definitely not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="Could not read index manifest"):
        load_manifest(manifest_path)


def test_invalid_manifest_schema_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"embedding_model": "bge-m3"}), encoding="utf-8")

    with pytest.raises(ConfigError, match="manifest.*invalid"):
        load_manifest(manifest_path)


def test_missing_manifest_with_preexisting_collection_is_not_trusted(tmp_path: Path) -> None:
    """A missing manifest must not make a pre-existing collection look compatible.

    ``assert_manifest_consistent`` alone passes when the local manifest is
    absent, so the *collection* itself must be inspected as well. A collection
    that already holds committed vectors with an unknown/legacy or mismatched
    model must be refused by ``assert_embedding_model_consistent``.
    """
    config = _config()
    manifest_path = tmp_path / "index" / "manifest.json"

    # No manifest on disk: the manifest-level check passes on its own...
    assert load_manifest(manifest_path) is None
    assert_manifest_consistent(manifest_path, config)

    # ...but the collection-level guard must still refuse legacy vectors.
    class LegacyCollection:
        def get(self, include=None):
            return {"ids": ["a", "b"], "metadatas": [{"embedding_model": None}, {}]}

    with pytest.raises(ValueError, match="no embedding_model metadata"):
        assert_embedding_model_consistent(LegacyCollection(), config.embedding.model)

    # A collection tagged with a *different* explicit model is refused too,
    # even though the manifest is absent.
    class DifferentModelCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "old-family"}],
            }

    with pytest.raises(ValueError, match="Mixing embedding spaces"):
        assert_embedding_model_consistent(
            DifferentModelCollection(), config.embedding.model
        )

    # A collection tagged with the configured model is consistent.
    class SameModelCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": config.embedding.model}],
            }

    assert_embedding_model_consistent(SameModelCollection(), config.embedding.model)


def test_auto_embedding_family_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("config/sources.example.yaml").read_text(encoding="utf-8"))
    data["embedding"]["model"] = "auto"
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="model: auto is not safe"):
        AppConfig.from_yaml(config_path)
