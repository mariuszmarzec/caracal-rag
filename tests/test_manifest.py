from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from caracal_rag.config import AppConfig, ConfigError
from caracal_rag.manifest import (
    assert_manifest_consistent,
    build_manifest,
    load_manifest,
    write_manifest,
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


def test_auto_embedding_family_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("config/sources.example.yaml").read_text(encoding="utf-8"))
    data["embedding"]["model"] = "auto"
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="model: auto is not safe"):
        AppConfig.from_yaml(config_path)
