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
from caracal_rag.vectorstore import (  # noqa: E402
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    collection_embedding_dim,
)


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


def test_manifest_persists_and_round_trips_embedding_dim(tmp_path: Path) -> None:
    config = _config()
    manifest_path = tmp_path / "index" / "manifest.json"

    write_manifest(
        manifest_path,
        build_manifest(config, document_count=4, embedding_dim=1024),
    )
    saved = load_manifest(manifest_path)
    assert saved is not None
    assert saved.embedding_dim == 1024
    assert saved.matches(config, observed_dim=1024)
    assert not saved.matches(config, observed_dim=512)


def test_legacy_manifest_without_dim_is_refused_when_probed(tmp_path: Path) -> None:
    """A manifest written before dimension tracking must not be trusted.

    The model, collection and api_base agree, but without an ``embedding_dim``
    the manifest cannot prove the embedding space, so it is refused when a
    live probe is available -- mirroring how a legacy Chroma collection is
    refused.
    """
    config = _config()
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, build_manifest(config, document_count=3))

    with pytest.raises(ConfigError, match="Mixing embedding spaces"):
        assert_manifest_consistent(manifest_path, config, observed_dim=1024)


def test_manifest_dimension_drift_is_refused(tmp_path: Path) -> None:
    config = _config()
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        build_manifest(config, document_count=3, embedding_dim=1024),
    )

    with pytest.raises(ConfigError, match="dimensional"):
        assert_manifest_consistent(manifest_path, config, observed_dim=512)


def test_invalid_manifest_embedding_dim_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "caracal-base",
                "embedding_model": "bge-m3",
                "embedding_api_base": "http://localhost:3001/v1",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "embedding_dim": "not-a-dimension",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="manifest.*invalid"):
        load_manifest(manifest_path)


def test_collection_embedding_dim_metadata_path() -> None:
    class MetadataCollection:
        def get(self, include=None):
            return {
                "ids": ["a", "b"],
                "metadatas": [
                    {"embedding_dim": 1024},
                    {"embedding_dim": 1024},
                ],
                "embeddings": [[0.0] * 1024, [0.0] * 1024],
            }

    assert collection_embedding_dim(MetadataCollection()) == 1024


def test_collection_embedding_dim_legacy_derives_from_vectors() -> None:
    class LegacyCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_model": "bge-m3"}],
                "embeddings": [[0.0] * 1024],
            }

    assert collection_embedding_dim(LegacyCollection()) == 1024


def test_collection_embedding_dim_empty_is_none() -> None:
    class EmptyCollection:
        def get(self, include=None):
            return {"ids": [], "metadatas": [], "embeddings": []}

    assert collection_embedding_dim(EmptyCollection()) is None


def test_collection_embedding_dim_mixed_is_refused() -> None:
    class MixedCollection:
        def get(self, include=None):
            return {
                "ids": ["a", "b"],
                "metadatas": [
                    {"embedding_dim": 1024},
                    {"embedding_dim": 512},
                ],
                "embeddings": [[0.0] * 1024, [0.0] * 512],
            }

    with pytest.raises(ValueError, match="dimensional embedding spaces"):
        collection_embedding_dim(MixedCollection())


def test_assert_embedding_dim_consistent_drift_is_refused() -> None:
    class DriftCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_dim": 1024}],
                "embeddings": [[0.0] * 1024],
            }

    with pytest.raises(ValueError, match="dimensional vectors"):
        assert_embedding_dim_consistent(DriftCollection(), 512)


def test_assert_embedding_dim_consistent_empty_passes() -> None:
    class EmptyCollection:
        def get(self, include=None):
            return {"ids": [], "metadatas": [], "embeddings": []}

    assert_embedding_dim_consistent(EmptyCollection(), 512)


def test_assert_embedding_dim_consistent_matching_passes() -> None:
    class MatchingCollection:
        def get(self, include=None):
            return {
                "ids": ["a"],
                "metadatas": [{"embedding_dim": 1024}],
                "embeddings": [[0.0] * 1024],
            }

    assert_embedding_dim_consistent(MatchingCollection(), 1024)
