"""Tests for endpoint-path normalisation and extraction.

These tests run without any external services: no LiteLLM, no GitHub, no Chroma.
"""

from __future__ import annotations

import pytest

from caracal_rag.endpoints import (
    endpoint_method,
    extract_endpoint,
    normalize_path,
    path_only,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/fiteo/api/1/tasks", "/fiteo/api/1/tasks"),
        ("/fiteo/api/1/tasks/", "/fiteo/api/1/tasks"),
        ("`/fiteo/api/1/tasks`", "/fiteo/api/1/tasks"),
        ("/fiteo//api/1//tasks", "/fiteo/api/1/tasks"),
        ("/", "/"),
        ("  /fiteo/api/1/tasks  ", "/fiteo/api/1/tasks"),
        ("/fiteo/api/1/tasks'", "/fiteo/api/1/tasks"),
    ],
)
def test_normalize_path(raw: str, expected: str) -> None:
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/fiteo/api/1/tasks", "/fiteo/api/1/tasks"),
        ("GET /fiteo/api/1/tasks", "/fiteo/api/1/tasks"),
        ("post /fiteo/api/1/tasks", "/fiteo/api/1/tasks"),
        ("  POST  /fiteo/api/1/tasks  ", "/fiteo/api/1/tasks"),
        ("/fiteo/api/1/tasks/", "/fiteo/api/1/tasks"),
    ],
)
def test_path_only(raw: str, expected: str) -> None:
    assert path_only(raw) == expected


def test_path_only_root() -> None:
    assert path_only("/") == "/"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("GET /fiteo/api/1/tasks", "GET"),
        ("post /fiteo/api/1/tasks", "POST"),
        ("/fiteo/api/1/tasks", None),
    ],
)
def test_endpoint_method(raw: str, expected: str | None) -> None:
    assert endpoint_method(raw) == expected


def test_extract_endpoint_from_heading() -> None:
    text = "# GET /fiteo/api/1/tasks\n\nGets all tasks.\n"
    endpoint = extract_endpoint(text)
    assert endpoint is not None
    assert endpoint.method == "GET"
    assert endpoint.path == "/fiteo/api/1/tasks"
    assert endpoint.normalized == "GET /fiteo/api/1/tasks"


def test_extract_endpoint_normalises_path() -> None:
    text = "# POST /fiteo/api/1/login/\n\nLogs in.\n"
    endpoint = extract_endpoint(text)
    assert endpoint is not None
    assert endpoint.path == "/fiteo/api/1/login"


def test_extract_endpoint_returns_none_when_no_method() -> None:
    text = "# API Documentation\n\nSome prose without an endpoint heading.\n"
    assert extract_endpoint(text) is None


def test_extract_endpoint_ignores_non_http_verbs() -> None:
    # "GET" is a real HTTP verb, but "NOTE" is not — it must not be mistaken for one.
    text = "# NOTE /fiteo/api/1/tasks\n\nThis is a note.\n"
    assert extract_endpoint(text) is None


def test_extract_endpoint_falls_back_to_inline_pair() -> None:
    text = "See `GET /fiteo/api/1/tasks` for details.\n"
    endpoint = extract_endpoint(text)
    assert endpoint is not None
    assert endpoint.method == "GET"
    assert endpoint.path == "/fiteo/api/1/tasks"


def test_extract_endpoint_inline_pair_requires_leading_slash() -> None:
    # An inline "GET tasks" without a leading slash is not an endpoint path.
    text = "GET tasks are listed here.\n"
    assert extract_endpoint(text) is None
