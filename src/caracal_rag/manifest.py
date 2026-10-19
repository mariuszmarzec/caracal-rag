"""Persisted index manifest under ``~/.caracal/index/manifest.json``.

The manifest records the embedding family/model that produced the stored
vectors, so the configured model can be compared against the *actual* index
before any indexing or search runs. It is complementary to the per-chunk
``embedding_model`` metadata in Chroma: the manifest is the durable record
written by the installer/indexer under the user's home, while the Chroma
metadata travels with the vectors. Both must agree; if the configured model
changes, the old vectors are never silently reused or mixed with new ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caracal_rag.config import AppConfig, ConfigError


@dataclass(frozen=True)
class IndexManifest:
    """Record of the embedding space used to build the active index."""

    collection: str
    embedding_model: str
    embedding_api_base: str
    updated_at: str
    document_count: int = 0
    #: Dimensionality of the vectors stored in the index. ``None`` for manifests
    #: written before dimension tracking existed; such a manifest cannot prove
    #: the embedding space and is refused by :func:`assert_manifest_consistent`
    #: when a live dimension probe is available.
    embedding_dim: int | None = None

    def matches(self, config: AppConfig, observed_dim: int | None = None) -> bool:
        """Return True when this manifest agrees with the active configuration.

        When ``observed_dim`` is supplied (the live probe of the configured
        model), the manifest must also record the same dimension. A manifest
        that predates dimension tracking (``embedding_dim`` is ``None``) cannot
        prove the space and is treated as a mismatch, mirroring how a legacy
        Chroma collection is refused.
        """
        if not (
            self.collection == config.chroma.collection
            and self.embedding_model == config.embedding.model
            and self.embedding_api_base == config.embedding.api_base
        ):
            return False
        if observed_dim is None:
            return True
        if self.embedding_dim is None:
            return False
        return self.embedding_dim == observed_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        required = ("collection", "embedding_model", "embedding_api_base", "updated_at")
        missing = [key for key in required if not str(data.get(key) or "").strip()]
        if missing:
            raise ValueError(
                "Index manifest is missing required field(s): "
                + ", ".join(missing)
            )
        embedding_dim: int | None = None
        dim = data.get("embedding_dim")
        if dim is not None:
            if isinstance(dim, bool) or isinstance(dim, float):
                raise ValueError("embedding_dim must be a positive integer")
            if isinstance(dim, int):
                embedding_dim = dim
            elif isinstance(dim, str) and dim.strip().lstrip("-").isdigit():
                embedding_dim = int(dim)
            else:
                raise ValueError("embedding_dim must be a positive integer")
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be a positive integer")
        return cls(
            collection=str(data["collection"]),
            embedding_model=str(data["embedding_model"]),
            embedding_api_base=str(data["embedding_api_base"]),
            updated_at=str(data["updated_at"]),
            document_count=int(data.get("document_count") or 0),
            embedding_dim=embedding_dim,
        )


def load_manifest(path: str | Path) -> IndexManifest | None:
    """Load the manifest, returning ``None`` when it is absent or unreadable.

    A missing manifest is not an error: it simply means no index has been
    recorded yet, so the first indexing run establishes the record.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Could not read index manifest at {path}: {exc}. Remove it and "
            "re-run indexing to recreate it."
        ) from exc
    try:
        return IndexManifest.from_dict(data)
    except ValueError as exc:
        raise ConfigError(
            f"Index manifest at {path} is invalid: {exc}. Remove it and "
            "re-run indexing to recreate it."
        ) from exc


def write_manifest(path: str | Path, manifest: IndexManifest) -> None:
    """Write the manifest atomically with restrictive permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)
    try:
        path.chmod(0o644)
    except OSError:
        pass


def assert_manifest_consistent(
    manifest_path: str | Path,
    config: AppConfig,
    observed_dim: int | None = None,
) -> None:
    """Raise ``ConfigError`` when a manifest disagrees with the active config.

    A mismatch means the configured embedding family/model changed since the
    index was built. The old vectors live in a different embedding space and
    must never be silently reused or mixed, so fail with rebuild guidance
    instead.

    When ``observed_dim`` is supplied (the live probe of the configured model),
    the manifest must also record the same dimension. A manifest that predates
    dimension tracking (``embedding_dim`` is ``None``) cannot prove the
    embedding space and is refused even when collection/model/api_base agree,
    mirroring how a legacy Chroma collection is refused.
    """
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return
    if manifest.matches(config, observed_dim=observed_dim):
        return
    if manifest.embedding_dim is not None and observed_dim is not None and (
        manifest.embedding_model == config.embedding.model
        and manifest.embedding_api_base == config.embedding.api_base
    ):
        raise ConfigError(
            f"Active embedding model {config.embedding.model!r} produces "
            f"{observed_dim}-dimensional vectors but the index at "
            f"{config.chroma.collection!r} was built with "
            f"{manifest.embedding_dim}-dimensional vectors. The stored vectors "
            "live in a different embedding space and must never be mixed with "
            "query vectors from the configured model. Rebuild the index with the "
            "current model (caracal-rag index) or point the configuration at a "
            "separate collection."
        )
    raise ConfigError(
        f"Active embedding model is {config.embedding.model!r} but the index at "
        f"{config.chroma.collection!r} was built with "
        f"{manifest.embedding_model!r}. Mixing embedding spaces produces "
        "meaningless search results. Rebuild the index with the current model "
        "(caracal-rag index) or point the configuration at a separate collection."
    )


def build_manifest(
    config: AppConfig,
    document_count: int,
    embedding_dim: int | None = None,
) -> IndexManifest:
    """Build a fresh manifest reflecting the active configuration."""
    return IndexManifest(
        collection=config.chroma.collection,
        embedding_model=config.embedding.model,
        embedding_api_base=config.embedding.api_base,
        updated_at=datetime.now(timezone.utc).isoformat(),
        document_count=document_count,
        embedding_dim=embedding_dim,
    )

