"""Tests for the required access-token authentication.

Every protected interface path must authenticate by default; missing or
invalid tokens are denied and no silent unauthenticated fallback exists. The
configured token is never echoed in errors or logs, so these tests also cover
secret redaction.
"""

from __future__ import annotations

import sys
import types

import pytest

# Stub out heavy optional dependencies so importing caracal_rag.auth works in CI.
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

from caracal_rag.auth import AuthenticationError, authenticate, redact_token  # noqa: E402
from caracal_rag.config import AccessTokenConfig, ConfigError  # noqa: E402

VALID_TOKEN = "correct-horse-battery-staple"


def _config(*, enabled: bool = True, token: str | None = VALID_TOKEN) -> AccessTokenConfig:
    return AccessTokenConfig(enabled=enabled, token=token)


def test_authenticate_accepts_valid_token() -> None:
    assert authenticate(_config(), VALID_TOKEN) is True


def test_authenticate_rejects_missing_token() -> None:
    with pytest.raises(AuthenticationError, match="no access token was presented"):
        authenticate(_config(), None)


def test_authenticate_rejects_empty_token() -> None:
    with pytest.raises(AuthenticationError, match="no access token was presented"):
        authenticate(_config(), "")


def test_authenticate_rejects_invalid_token() -> None:
    with pytest.raises(AuthenticationError, match="does not match"):
        authenticate(_config(), "wrong-token-value")


def test_authenticate_rejects_short_token() -> None:
    with pytest.raises(AuthenticationError, match="too short"):
        authenticate(_config(), "short")


def test_authenticate_rejects_short_configured_token() -> None:
    """A configured token below the minimum length is refused at runtime."""
    with pytest.raises(AuthenticationError, match="no access token is configured"):
        authenticate(AccessTokenConfig(enabled=True, token="short"), VALID_TOKEN)


def test_authenticate_no_token_when_disabled_is_allowed() -> None:
    """Disabling auth is an explicit opt-out in the active configuration."""
    assert authenticate(_config(enabled=False, token=None), None) is True


def test_authenticate_disabled_does_not_validate_presented_token() -> None:
    assert authenticate(_config(enabled=False, token=None), "anything-at-all") is True


def test_authentication_error_is_config_error() -> None:
    assert issubclass(AuthenticationError, ConfigError)


def test_error_messages_never_reveal_token() -> None:
    token = VALID_TOKEN
    for presented in (None, "", "short", "wrong-token-value"):
        with pytest.raises(AuthenticationError) as excinfo:
            authenticate(_config(token=token), presented)
        assert token not in str(excinfo.value)
        assert "***" not in str(excinfo.value)


def test_redact_token_replaces_occurrences() -> None:
    text = f"token={VALID_TOKEN} and again {VALID_TOKEN} end"
    redacted = redact_token(text, VALID_TOKEN)
    assert VALID_TOKEN not in redacted
    assert redacted.count("***") == 2


def test_redact_token_no_secret_is_noop() -> None:
    text = "nothing to redact here"
    assert redact_token(text, None) == text
    assert redact_token(text, "") == text


def test_compare_digest_is_used_for_constant_time_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparison must go through ``secrets.compare_digest`` so a valid/
    invalid comparison does not leak timing information about the configured
    token."""
    calls: list[tuple] = []

    def fake_compare_digest(a: str, b: str) -> bool:
        calls.append((a, b))
        return a == b

    import caracal_rag.auth as auth

    monkeypatch.setattr(auth.secrets, "compare_digest", fake_compare_digest)
    assert authenticate(_config(token=VALID_TOKEN), VALID_TOKEN) is True
    assert calls == [(VALID_TOKEN, VALID_TOKEN)]

    calls.clear()
    with pytest.raises(AuthenticationError):
        authenticate(_config(token=VALID_TOKEN), "wrong-token-value")
    assert calls == [(VALID_TOKEN, "wrong-token-value")]
