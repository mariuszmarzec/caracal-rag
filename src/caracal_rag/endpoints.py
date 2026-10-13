"""Endpoint path extraction and lookup helpers.

API documentation chunks are indexed as plain Markdown text. To support exact
endpoint-path lookup we extract the HTTP method + path from each chunk's text
and normalise it so that ``find_endpoint("/fiteo/api/1/tasks")``,
``find_endpoint("GET /fiteo/api/1/tasks")`` and ``find_endpoint("fiteo/api/1/tasks")``
all resolve to the same indexed document.

The extraction is deliberately conservative: it only recognises paths that are
explicitly declared in the document (the first Markdown heading of the form
``METHOD /path``). It never fabricates matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches a Markdown heading like:  "GET /fiteo/api/1/tasks" or "# POST /path"
_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<method>[A-Z]+)\s+(?P<path>\S+)", re.MULTILINE)
# Matches an inline method/path pair like: "GET /fiteo/api/1/tasks" anywhere.
_INLINE_RE = re.compile(r"\b(?P<method>[A-Z]{2,7})\s+(?P<path>/[^\s`)\]]+)")

_HTTP_METHODS = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "HEAD",
    "OPTIONS",
    "CONNECT",
    "TRACE",
}


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str

    @property
    def normalized(self) -> str:
        """Return the canonical lookup key for this endpoint."""
        return f"{self.method.upper()} {normalize_path(self.path)}"


def normalize_path(path: str) -> str:
    """Normalise an endpoint path for lookup.

    - Strips surrounding whitespace and surrounding quotes/backticks.
    - Collapses repeated slashes.
    - Removes a trailing slash (except for the root ``/``).
    """
    cleaned = path.strip().strip("`'\"").strip()
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned


def extract_endpoint(text: str) -> Endpoint | None:
    """Extract the first explicit HTTP method + path from a chunk of text.

    Returns ``None`` when no explicit endpoint declaration is found, so callers
    can distinguish "no match" from "endpoint found".
    """
    match = _HEADING_RE.search(text)
    if match and match.group("method") in _HTTP_METHODS:
        return Endpoint(method=match.group("method"), path=normalize_path(match.group("path")))

    for match in _INLINE_RE.finditer(text):
        method = match.group("method")
        if method not in _HTTP_METHODS:
            continue
        path = normalize_path(match.group("path"))
        return Endpoint(method=method, path=path)

    return None


def path_only(path: str) -> str:
    """Return only the normalised path portion of a user-supplied lookup.

    Accepts either a bare path (``/fiteo/api/1/tasks``) or a method + path
    pair (``GET /fiteo/api/1/tasks``).
    """
    cleaned = path.strip()
    parts = cleaned.split()
    if len(parts) >= 2 and parts[0].upper() in _HTTP_METHODS:
        return normalize_path(parts[1])
    return normalize_path(cleaned)


def endpoint_method(path: str) -> str | None:
    """Return the method portion of a user-supplied lookup, if any."""
    parts = path.strip().split()
    if parts and parts[0].upper() in _HTTP_METHODS:
        return parts[0].upper()
    return None
