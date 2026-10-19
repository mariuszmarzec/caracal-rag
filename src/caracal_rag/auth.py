"""Required access-token authentication for protected Caracal RAG interfaces.

Every interface that should be protected (the MCP server, and any future
HTTP/RAG endpoint) must authenticate the caller through a configured access
token. There is no silent unauthenticated fallback: a missing or invalid
token is denied with a clear error that never reveals the configured token or
any other secret.

The token itself is never compared into logs, diagnostics, exceptions, or
tool-argument echoes. :func:`authenticate` returns a plain ``bool`` and raises
:class:`caracal_rag.config.ConfigError` on failure, so callers can surface an
actionable message without ever handling the raw secret.

The HTTP transport boundary uses :class:`SimpleTokenVerifier`, which plugs
into the MCP SDK's ``TokenVerifier`` protocol so the server can demand a
``Authorization: Bearer <token>`` header on every request without the secret
ever appearing in a model-visible tool schema, tool argument, tool result, or
LLM-facing transcript.
"""

from __future__ import annotations

import secrets
from typing import Any

from caracal_rag.config import AccessTokenConfig, ConfigError


class AuthenticationError(ConfigError):
    """Raised when a protected interface rejects a missing or invalid token.

    Inherits from :class:`ConfigError` so existing ``except ConfigError``
    handlers (CLI, MCP) surface it with the same actionable guidance, while
    remaining distinguishable for tests and interface code that wants a
    distinct auth-failure path.
    """


def _token_shape_ok(token: str | None) -> bool:
    """Return True when ``token`` is a plausible configured secret.

    A configured token must be a non-empty string of at least
    :data:`~caracal_rag.config.MIN_TOKEN_LENGTH` characters. Short placeholder
    values are refused at runtime so a half-configured deployment cannot
    silently authenticate.
    """
    if not isinstance(token, str):
        return False
    return len(token) >= 8


def is_authentication_required(access_token: AccessTokenConfig) -> bool:
    """Return True when the protected interface must authenticate.

    ``access_token.enabled`` defaults to ``True``: every intended protected
    path requires a valid token by default. Disabling is an explicit opt-out
    that must be set in the active configuration, never inferred from an
    absent token.
    """
    return bool(access_token.enabled)


def authenticate(access_token: AccessTokenConfig, presented: str | None) -> bool:
    """Return True when ``presented`` matches the configured access token.

    Missing or invalid tokens are denied; there is no silent unauthenticated
    fallback. The configured token is never echoed: on failure the raised
    error names what to fix and where, not the secret itself.

    Comparison is constant-time via ``secrets.compare_digest`` when both sides
    are strings, so a valid/invalid comparison does not leak timing information
    about the configured value.
    """
    if not is_authentication_required(access_token):
        return True
    configured = access_token.token
    if not _token_shape_ok(configured):
        raise AuthenticationError(
            "Access authentication is enabled but no access token is configured. "
            "Set it by running 'caracal-rag setup', by exporting "
            "CARACAL_ACCESS_TOKEN, or by setting `access_token.token` in the "
            "secrets file under your Caracal home "
            "(~/.caracal/config/secrets.yaml)."
        )
    if not isinstance(presented, str) or not presented:
        raise AuthenticationError(
            "Access denied: no access token was presented. Protected Caracal RAG "
            "interfaces require a valid token. Supply it as the `access_token` "
            "field on the request (for example "
            "search_api({\"query\": \"...\", \"access_token\": \"...\"})), or set "
            "CARACAL_ACCESS_TOKEN in the MCP server's environment."
        )
    if not _token_shape_ok(presented):
        raise AuthenticationError(
            "Access denied: the presented token is too short to be a valid "
            "credential. Supply a token of at least 8 characters."
        )

    if not secrets.compare_digest(configured, presented):
        raise AuthenticationError(
            "Access denied: the presented access token does not match the "
            "configured token. Rotate the token with 'caracal-rag setup' and "
            "update your client configuration; the configured value is never "
            "shown by this error."
        )
    return True


def redact_token(text: str, token: str | None) -> str:
    """Return ``text`` with any occurrence of ``token`` replaced by ``***``.

    Mirrors :func:`caracal_rag.embeddings.redact_secret` so diagnostics and
    error messages can never echo the access token.
    """
    if not token:
        return text
    return text.replace(token, "***")


class SimpleTokenVerifier:
    """Token verifier for the MCP SDK's HTTP auth boundary.

    The MCP SDK validates bearer tokens through the
    :class:`mcp.server.auth.provider.TokenVerifier` protocol. This adapter
    wraps the configured access-token policy so the HTTP transport can demand
    ``Authorization: Bearer <token>`` on every request. The presented token is
    consumed from the HTTP request header, never from a model-generated tool
    argument, so the secret never reaches the model-visible tool schema, tool
    arguments, tool results, or LLM-facing transcript.

    The verifier is intentionally not constant-time on the *absence* of a
    token: a missing header is a cheap reject that does not touch the secret.
    When a token is actually presented, comparison still goes through
    :func:`authenticate`, which uses ``secrets.compare_digest``.
    """

    def __init__(self, access_token: AccessTokenConfig) -> None:
        self.access_token = access_token

    async def verify_token(self, token: str) -> Any:
        """Return an :class:`AccessToken` when ``token`` is valid, else ``None``.

        Returning ``None`` tells the SDK's ``BearerAuthBackend`` that the
        request is unauthenticated, which the surrounding
        ``RequireAuthMiddleware`` turns into a ``401``. The configured token is
        never echoed in the returned object or in any error.
        """
        # Local import: the MCP SDK is an optional dependency at the auth
        # module level, and this verifier is only used by the HTTP transport
        # path that imports it explicitly.
        from mcp.server.auth.provider import AccessToken

        try:
            authenticate(self.access_token, token)
        except AuthenticationError:
            return None
        return AccessToken(token=token, client_id="caracal-rag", scopes=[])
