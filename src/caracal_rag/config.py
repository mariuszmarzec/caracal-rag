from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Environment variable that overrides the Caracal home directory.
CARACAL_HOME_ENV = "CARACAL_HOME"
#: Environment variable with a FreeLLMAPI key; takes precedence over secrets.yaml.
EMBEDDING_API_KEY_ENV = "CARACAL_EMBEDDING_API_KEY"

DEFAULT_EMBEDDING_API_BASE = "http://localhost:3001/v1"


class ConfigError(Exception):
    """Configuration or credential setup problem with actionable guidance.

    The message must never contain a secret value; callers redact before
    printing.
    """


@dataclass(frozen=True)
class CaracalHome:
    """Resolved ``~/.caracal`` layout for one user."""

    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "config" / "sources.yaml"

    @property
    def secrets_path(self) -> Path:
        return self.root / "config" / "secrets.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "index" / "manifest.json"


def resolve_home(override: str | Path | None = None) -> CaracalHome:
    """Resolve the Caracal home directory.

    Priority: explicit override > ``$CARACAL_HOME`` > ``~/.caracal``. The home
    directory is resolved from ``Path.home()`` (i.e. ``$HOME``), never from the
    current working directory.
    """
    if override is not None:
        root = Path(override).expanduser()
    else:
        env = os.getenv(CARACAL_HOME_ENV)
        root = Path(env).expanduser() if env else Path.home() / ".caracal"
    return CaracalHome(root)


@dataclass
class SourceConfig:
    name: str
    type: str
    urls: list[str] | None = None
    url: str | None = None
    repo: str | None = None
    path: str | None = None
    branch: str | None = None

    def __post_init__(self) -> None:
        has_urls = self.urls or self.url
        if self.type == "github_md_doc_dir":
            if not self.repo or not self.path:
                raise ValueError(
                    f"Source {self.name!r} of type github_md_doc_dir must define repo and path"
                )
        elif self.type == "local":
            if not self.path:
                raise ValueError(
                    f"Source {self.name!r} of type local must define path"
                )
        elif not has_urls:
            raise ValueError(
                f"Source {self.name!r} must define urls, url, or repo+path"
            )


@dataclass
class EmbeddingConfig:
    #: Explicit FreeLLMAPI embedding family/model. There is intentionally no
    #: built-in default: silently choosing a model would let an index be built
    #: in an embedding space the user did not pick.
    model: str
    api_base: str = field(
        default_factory=lambda: os.getenv("CARACAL_EMBEDDING_API_BASE", DEFAULT_EMBEDDING_API_BASE)
    )
    api_key: str | None = field(
        default_factory=lambda: os.getenv(EMBEDDING_API_KEY_ENV),
    )


@dataclass
class ChromaConfig:
    host: str = field(default_factory=lambda: os.getenv("CARACAL_CHROMA_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("CARACAL_CHROMA_PORT", "8000")))
    collection: str = field(
        default_factory=lambda: os.getenv("CARACAL_CHROMA_COLLECTION", "caracal")
    )
    ssl: bool = field(
        default_factory=lambda: os.getenv("CARACAL_CHROMA_SSL", "false").lower()
        in {"1", "true", "yes"}
    )


def _coerce_source(item: dict[str, Any]) -> SourceConfig:
    """Coerce a raw source dict into a SourceConfig."""
    return SourceConfig(**item)


@dataclass
class AppConfig:
    sources: list[SourceConfig]
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    #: Where this configuration was loaded from, when known. Runtime commands
    #: use it to load documents from the *active* configuration instead of a
    #: checkout-relative path.
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        sources = [_coerce_source(item) for item in data.get("sources", [])]
        embedding_data = dict(data.get("embedding", {}) or {})
        chroma_data = data.get("chroma", {}) or {}

        if not str(embedding_data.get("model") or "").strip():
            raise ConfigError(
                f"embedding.model is missing in {path}. Set an explicit "
                "FreeLLMAPI embedding family/model (for example "
                "`embedding.model: bge-m3`) after checking what is enabled in "
                "your deployment under FreeLLMAPI Models -> Embeddings, or "
                "export CARACAL_EMBEDDING_MODEL. There is no built-in default "
                "model: a silently chosen family would make the index "
                "incompatible with the vectors you expect."
            )
        embedding = EmbeddingConfig(**embedding_data)
        chroma = ChromaConfig(**chroma_data)

        return cls(sources=sources, embedding=embedding, chroma=chroma, config_path=path)


def load_secrets(path: Path) -> dict[str, str]:
    """Load user secrets (the FreeLLMAPI key) from the secrets file.

    The file must not be readable by group or other users; a loose mode is
    refused rather than silently read, because the key must never sit in a
    world-readable file. Returns an empty dict when the file does not exist.
    """
    if not path.is_file():
        return {}
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ConfigError(
            f"Secrets file {path} is group/world-readable (mode {oct(mode & 0o777)}). "
            f"Refusing to read it. Fix it with: chmod 600 {path}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    embedding_data = data.get("embedding", {}) or {}
    api_key = embedding_data.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return {"api_key": api_key}
    return {}


def write_secret_api_key(path: Path, api_key: str) -> None:
    """Persist the FreeLLMAPI key in the secrets file with mode 0600.

    Any other keys already present in the file are preserved. The file is
    created with 0600 regardless of umask and chmod'd afterwards in case it
    pre-existed with a looser mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    embedding_data = dict(data.get("embedding", {}) or {})
    embedding_data["api_key"] = api_key
    data["embedding"] = embedding_data
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    os.chmod(path, 0o600)


def write_config(config: AppConfig, path: Path) -> None:
    """Write the active configuration (never including the API key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "embedding": {
            "model": config.embedding.model,
            "api_base": config.embedding.api_base,
        },
        "chroma": {
            "host": config.chroma.host,
            "port": config.chroma.port,
            "collection": config.chroma.collection,
            "ssl": config.chroma.ssl,
        },
        "sources": [_source_to_dict(source) for source in config.sources],
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def _source_to_dict(source: SourceConfig) -> dict[str, Any]:
    item: dict[str, Any] = {"name": source.name, "type": source.type}
    for key in ("urls", "url", "repo", "path", "branch"):
        value = getattr(source, key)
        if value is not None:
            item[key] = value
    return item


def load_active_config(config_path: str | Path | None = None) -> AppConfig:
    """Load the active installed configuration and credentials.

    Resolution order for the configuration file: explicit ``config_path`` >
    ``$CARACAL_HOME/config/sources.yaml`` (default ``~/.caracal``). Secrets
    always come from the home's ``config/secrets.yaml`` (mode-checked) plus the
    ``CARACAL_EMBEDDING_API_KEY`` environment variable, which takes
    precedence. Fails early with actionable guidance when the active
    configuration is missing or the secrets file has unsafe permissions.
    """
    home = resolve_home()
    if config_path is not None:
        path = Path(config_path).expanduser()
    else:
        path = home.config_path
    if not path.is_file():
        raise ConfigError(
            f"No active configuration found at {path}. Install Caracal with "
            "install.sh (it creates the configuration under "
            f"{home.root}), or run 'caracal-rag setup' to create it. Runtime "
            "commands always use this active configuration, never a "
            "checkout-relative file."
        )
    config = AppConfig.from_yaml(path)
    secrets = load_secrets(home.secrets_path)
    if secrets.get("api_key"):
        config.embedding.api_key = secrets["api_key"]
    env_key = os.getenv(EMBEDDING_API_KEY_ENV)
    if env_key:
        config.embedding.api_key = env_key
    return config
