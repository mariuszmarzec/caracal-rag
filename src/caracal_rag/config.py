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
#: Environment variable with the Caracal RAG access token; takes precedence over
#: secrets.yaml. Used by every protected interface (the MCP server). It is read
#: from the server's own environment so an MCP client never has to smuggle a
#: secret through a tool-argument field.
ACCESS_TOKEN_ENV = "CARACAL_ACCESS_TOKEN"

DEFAULT_EMBEDDING_API_BASE = "http://localhost:3001/v1"

#: Minimum length for a configured access token. A shorter value is refused
#: during setup and at runtime instead of being silently accepted.
MIN_TOKEN_LENGTH = 8


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


#: Fields this version knows how to coerce into ``EmbeddingConfig``. Any other
#: key in the active YAML is preserved verbatim by ``write_config``.
_EMBEDDING_FIELDS = {"model", "api_base", "api_key"}
#: Fields this version knows how to coerce into ``ChromaConfig``.
_CHROMA_FIELDS = {"host", "port", "collection", "ssl"}

#: Fields this version knows how to coerce into ``AccessTokenConfig``.
_ACCESS_TOKEN_FIELDS = {"enabled", "token"}


@dataclass
class AccessTokenConfig:
    """Configuration for the required Caracal RAG access token.

    ``enabled`` defaults to ``True``: every protected interface path must
    authenticate by default, and there is no silent unauthenticated fallback.
    ``token`` is never stored in the active configuration file; it lives in
    ``secrets.yaml`` (mode ``0600``) or in the ``CARACAL_ACCESS_TOKEN``
    environment variable, in that order of precedence.
    """

    enabled: bool = True
    token: str | None = None


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
    """Coerce the fields this version understands, leaving unknown YAML untouched.

    Unknown source keys are ignored at runtime but are preserved by write_config
    when setup updates the active YAML file. This allows forward-compatible
    extensions without making a routine credential rotation erase user data.
    """
    allowed = {"name", "type", "urls", "url", "repo", "path", "branch"}
    return SourceConfig(**{key: value for key, value in item.items() if key in allowed})


@dataclass
class AppConfig:
    sources: list[SourceConfig]
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    access_token: AccessTokenConfig = field(
        default_factory=AccessTokenConfig
    )
    #: Where this configuration was loaded from, when known. Runtime commands
    #: use it to load documents from the *active* configuration instead of a
    #: checkout-relative path.
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Could not read active configuration at {path}; it was left unchanged. "
                "Repair the file or restore a valid backup before running setup again."
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Active configuration at {path} is not a YAML mapping; it was left unchanged."
            )

        sources = [_coerce_source(item) for item in data.get("sources", [])]
        embedding_data = dict(data.get("embedding", {}) or {})
        chroma_data = data.get("chroma", {}) or {}
        access_token_data = data.get("access_token", {}) or {}

        configured_model = embedding_data.get("model") or os.getenv("CARACAL_EMBEDDING_MODEL")
        if not str(configured_model or "").strip():
            raise ConfigError(
                f"embedding.model is missing in {path}. Set an explicit "
                "FreeLLMAPI embedding family/model (for example "
                "`embedding.model: bge-m3`) after checking what is enabled in "
                "your deployment under FreeLLMAPI Models -> Embeddings, or "
                "export CARACAL_EMBEDDING_MODEL. There is no built-in default "
                "model: a silently chosen family would make the index "
                "incompatible with the vectors you expect."
            )
        if str(configured_model).strip().lower() == "auto":
            raise ConfigError(
                "embedding.model: auto is not safe for a persistent index because "
                "FreeLLMAPI may change its default family. Select an explicit "
                "family/model under FreeLLMAPI Models -> Embeddings and configure "
                "that value in embedding.model before indexing or serving MCP."
            )
        embedding_data["model"] = str(configured_model).strip()
        # Only pass keys this version understands to the dataclass. Unknown
        # user-defined keys (forward-compatible extensions) are kept in
        # ``embedding_data`` so write_config re-emits them verbatim instead of
        # discarding them or crashing on an unexpected keyword argument.
        embedding = EmbeddingConfig(
            **{key: value for key, value in embedding_data.items() if key in _EMBEDDING_FIELDS}
        )
        chroma = ChromaConfig(
            **{key: value for key, value in chroma_data.items() if key in _CHROMA_FIELDS}
        )
        access_token = AccessTokenConfig(
            **{
                key: value
                for key, value in access_token_data.items()
                if key in _ACCESS_TOKEN_FIELDS
            }
        )

        return cls(
            sources=sources,
            embedding=embedding,
            chroma=chroma,
            access_token=access_token,
            config_path=path,
        )


def load_secrets(path: Path) -> dict[str, str]:
    """Load user secrets (the FreeLLMAPI key and the access token) from the secrets file.

    The file must not be readable by group or other users; a loose mode is
    refused rather than silently read, because the key must never sit in a
    world-readable file. Returns an empty dict when the file does not exist.

    The returned mapping may carry ``api_key`` (from ``embedding.api_key``) and
    ``access_token`` (from ``access_token.token``). Either may be absent; the
    caller decides what is required for its interface.
    """
    if not path.is_file():
        return {}
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ConfigError(
            f"Secrets file {path} is group/world-readable (mode {oct(mode & 0o777)}). "
            f"Refusing to read it. Fix it with: chmod 600 {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"Could not read secrets file at {path}; the key was not loaded. "
            "Repair the YAML or restore a valid backup. Secret values are omitted from this error."
        ) from exc
    result: dict[str, str] = {}
    embedding_data = data.get("embedding", {}) or {}
    api_key = embedding_data.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        result["api_key"] = api_key
    token_data = data.get("access_token", {}) or {}
    token = token_data.get("token")
    if isinstance(token, str) and token.strip():
        result["access_token"] = token
    return result


def write_secret_api_key(path: Path, api_key: str) -> None:
    """Persist the FreeLLMAPI key in a private secrets file.

    Any other keys already present in the file are preserved. Existing files
    are made private before they are read or rewritten, and the open descriptor
    is forced to mode 0600 before secret bytes are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        os.chmod(path, 0o600)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Could not read secrets file at {path}; it was left unchanged. "
                "Repair the YAML or restore a valid backup before rotating credentials."
            ) from exc
    embedding_data = dict(data.get("embedding", {}) or {})
    embedding_data["api_key"] = api_key
    data["embedding"] = embedding_data
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def write_secret_access_token(path: Path, token: str) -> None:
    """Persist the Caracal RAG access token in the private secrets file.

    Mirrors :func:`write_secret_api_key`: any other keys already present in the
    file are preserved, existing files are made private before they are read or
    rewritten, and the open descriptor is forced to mode 0600 before secret
    bytes are written. The token is stored under ``access_token.token`` so it
    stays out of the active configuration file, which is intentionally
    secret-free.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        os.chmod(path, 0o600)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Could not read secrets file at {path}; it was left unchanged. "
                "Repair the YAML or restore a valid backup before rotating credentials."
            ) from exc
    token_data = dict(data.get("access_token", {}) or {})
    token_data["token"] = token
    data["access_token"] = token_data
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def write_config(config: AppConfig, path: Path) -> None:
    """Persist active settings while preserving existing user-owned YAML keys.

    Setup only changes the embedding model and endpoint. Existing source entries,
    custom extensions, and unknown configuration fields are retained verbatim
    semantically; an unreadable or non-mapping existing config is never replaced
    with defaults.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Could not read existing configuration at {path}; it was left unchanged. "
                "Repair the YAML manually before running setup again."
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Existing configuration at {path} is not a YAML mapping; it was left unchanged."
            )
        data = dict(data)
        embedding_data = data.get("embedding", {}) or {}
        if not isinstance(embedding_data, dict):
            raise ConfigError(
                f"The embedding section in {path} is not a mapping; the file was left unchanged."
            )
        embedding_data = dict(embedding_data)
        # Migrate any legacy key embedded in sources.yaml into secrets.yaml;
        # active config is deliberately secret-free.
        embedding_data.pop("api_key", None)
        embedding_data["model"] = config.embedding.model
        embedding_data["api_base"] = config.embedding.api_base
        data["embedding"] = embedding_data
        # The active configuration never holds the token; only the access_token
        # *policy* (enabled/disabled) is persisted here. Any legacy
        # ``access_token.token`` that leaked into the active config is migrated
        # into secrets.yaml by setup and stripped from this file.
        access_token_data = data.get("access_token", {}) or {}
        if isinstance(access_token_data, dict):
            access_token_data = dict(access_token_data)
            access_token_data.pop("token", None)
            access_token_data["enabled"] = config.access_token.enabled
            data["access_token"] = access_token_data
    else:
        data = {
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
            "access_token": {"enabled": config.access_token.enabled},
            "sources": [_source_to_dict(source) for source in config.sources],
        }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    tmp.replace(path)


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
    if secrets.get("access_token"):
        config.access_token.token = secrets["access_token"]
    env_token = os.getenv(ACCESS_TOKEN_ENV)
    if env_token:
        config.access_token.token = env_token
    return config
