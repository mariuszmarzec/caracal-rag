"""Tests for source configuration and API-doc discovery.

External services (GitHub, requests) are mocked with ``unittest.mock`` so the
tests run without network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from caracal_rag.config import SourceConfig
from caracal_rag.sources import (
    Document,
    SourceLoadError,
    _github_list_md_files,
    github_md_doc_dir_documents,
    infer_type,
    load_documents_from_source,
)


def _make_response(json_data, links=None):
    response = MagicMock()
    response.json.return_value = json_data
    response.links = links or {}
    response.raise_for_status.return_value = None
    return response


def test_github_list_md_files_flat_dir():
    contents = [
        {"type": "file", "name": "login.md", "path": "docs/api/login.md"},
        {"type": "file", "name": "config.json", "path": "docs/api/config.json"},
        {"type": "file", "name": "readme.md", "path": "docs/api/readme.md"},
    ]
    mock_response = _make_response(contents)
    with patch("caracal_rag.sources.requests.get", return_value=mock_response):
        files = list(_github_list_md_files("owner/repo", "docs/api", "master"))
    assert sorted(files) == ["docs/api/login.md", "docs/api/readme.md"]


def test_github_list_md_files_nested_dirs():
    root_contents = [
        {"type": "file", "name": "top.md", "path": "docs/api/top.md"},
        {"type": "dir", "name": "sub", "path": "docs/api/sub"},
    ]
    sub_contents = [
        {"type": "file", "name": "auth.md", "path": "docs/api/sub/auth.md"},
    ]
    responses = [_make_response(root_contents), _make_response(sub_contents)]
    with patch("caracal_rag.sources.requests.get", side_effect=responses):
        files = list(_github_list_md_files("owner/repo", "docs/api", "master"))
    assert sorted(files) == ["docs/api/sub/auth.md", "docs/api/top.md"]


def test_github_list_md_files_single_file_response():
    single = {"type": "file", "name": "only.md", "path": "docs/api/only.md"}
    mock_response = _make_response(single)
    with patch("caracal_rag.sources.requests.get", return_value=mock_response):
        files = list(_github_list_md_files("owner/repo", "docs/api/only.md", "master"))
    assert files == ["docs/api/only.md"]


def test_github_md_doc_dir_documents_fetches_content():
    listing = [
        {"type": "file", "name": "login.md", "path": "docs/api/login.md"},
    ]
    listing_response = _make_response(listing)
    raw_response = MagicMock()
    raw_response.text = "# POST /fiteo/api/1/login\n\nLogs in."
    raw_response.raise_for_status.return_value = None

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=[listing_response, raw_response],
    ):
        docs = list(
            github_md_doc_dir_documents(
                source_name="fiteo-docs",
                repo="mariuszmarzec/fiteo",
                path="docs/api",
                branch="master",
            )
        )

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "fiteo-docs"
    assert doc.name == "login.md"
    assert doc.type == "markdown"
    assert doc.url == (
        "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/login.md"
    )
    assert doc.content == "# POST /fiteo/api/1/login\n\nLogs in."
    assert doc.content_hash == Document.hash("# POST /fiteo/api/1/login\n\nLogs in.")


def test_load_documents_from_source_github_md_doc_dir():
    source = SourceConfig(
        name="fiteo",
        type="github_md_doc_dir",
        repo="mariuszmarzec/fiteo",
        path="docs/api",
        branch="master",
    )
    listing = [{"type": "file", "name": "login.md", "path": "docs/api/login.md"}]
    listing_response = _make_response(listing)
    raw_response = MagicMock()
    raw_response.text = "# Login"
    raw_response.raise_for_status.return_value = None

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=[listing_response, raw_response],
    ):
        docs = list(load_documents_from_source(source))

    assert len(docs) == 1
    assert docs[0].url == (
        "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/login.md"
    )


def test_load_documents_from_source_github_md_doc_dir_no_branch_defaults():
    source = SourceConfig(
        name="fiteo",
        type="github_md_doc_dir",
        repo="mariuszmarzec/fiteo",
        path="docs/api",
        branch=None,
    )
    listing_response = _make_response([])
    with patch("caracal_rag.sources.requests.get", return_value=listing_response):
        docs = list(load_documents_from_source(source))
    assert docs == []


def test_source_config_requires_repo_and_path():
    with pytest.raises(ValueError, match="must define repo and path"):
        SourceConfig(name="bad", type="github_md_doc_dir", repo="o/r", path=None)


def test_source_config_requires_urls_or_url_or_repo():
    with pytest.raises(ValueError, match="must define urls, url, or repo\\+path"):
        SourceConfig(name="bad", type="markdown", urls=None, url=None)


def test_infer_type_github_paths():
    assert infer_type("docs/api/login.md") == "markdown"
    assert infer_type("docs/api/data.json") == "json"
    assert infer_type("docs/api/config.yaml") == "yaml"


def test_load_documents_from_source_local(tmp_path: Path):
    docs_dir = tmp_path / "knowledge"
    docs_dir.mkdir()
    (docs_dir / "joke1.md").write_text("# Joke 1\n\nWhy did the chicken...")
    (docs_dir / "joke2.md").write_text("# Joke 2\n\nWhat do you call...")
    (docs_dir / "notmd.txt").write_text("This should be ignored")

    source = SourceConfig(
        name="funny-jokes",
        type="local",
        path=str(docs_dir),
    )

    docs = list(load_documents_from_source(source))
    assert len(docs) == 2
    names = {d.name for d in docs}
    assert "joke1.md" in names
    assert "joke2.md" in names
    assert all(d.type == "markdown" for d in docs)
    assert all(d.source == "funny-jokes" for d in docs)


def test_local_source_requires_path():
    with pytest.raises(ValueError, match="must define path"):
        SourceConfig(name="bad", type="local", path=None)


def test_config_indexes_fiteo_api_path():
    """The example config must explicitly point at docs/api on master."""
    from caracal_rag.config import AppConfig

    config = AppConfig.from_yaml("config/sources.example.yaml")
    fiteo = next(s for s in config.sources if s.name == "fiteo-docs")
    assert fiteo.type == "github_md_doc_dir"
    assert fiteo.repo == "mariuszmarzec/fiteo"
    assert fiteo.path == "docs/api"
    assert fiteo.branch == "master"


def test_github_md_doc_dir_documents_raises_on_fetch_failure():
    """A transient fetch failure raises SourceLoadError carrying loaded docs."""
    listing = [
        {"type": "file", "name": "ok.md", "path": "docs/api/ok.md"},
        {"type": "file", "name": "bad.md", "path": "docs/api/bad.md"},
    ]
    listing_response = _make_response(listing)
    ok_response = MagicMock()
    ok_response.text = "# OK"
    ok_response.raise_for_status.return_value = None
    bad_response = MagicMock()
    bad_response.raise_for_status.side_effect = Exception("connection reset")

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=[listing_response, ok_response, bad_response],
    ):
        with pytest.raises(SourceLoadError) as exc_info:
            list(
                github_md_doc_dir_documents(
                    source_name="fiteo-docs",
                    repo="mariuszmarzec/fiteo",
                    path="docs/api",
                    branch="master",
                )
            )

    assert len(exc_info.value.documents) == 1
    assert exc_info.value.documents[0].url == (
        "https://raw.githubusercontent.com/mariuszmarzec/fiteo/master/docs/api/ok.md"
    )


def test_github_md_doc_dir_documents_raises_on_discovery_failure():
    """A GitHub listing failure raises SourceLoadError with no loaded docs.

    Without this, a raw exception from the listing step aborts the run instead
    of being reported as a partial source load that the indexer can skip while
    preserving existing vectors.
    """
    listing_response = MagicMock()
    listing_response.raise_for_status.side_effect = Exception("rate limit")

    with patch(
        "caracal_rag.sources.requests.get",
        return_value=listing_response,
    ):
        with pytest.raises(SourceLoadError) as exc_info:
            list(
                github_md_doc_dir_documents(
                    source_name="fiteo-docs",
                    repo="mariuszmarzec/fiteo",
                    path="docs/api",
                    branch="master",
                )
            )

    assert exc_info.value.documents == []
    assert "Failed to list markdown files" in str(exc_info.value)


def test_load_documents_from_source_raises_on_fetch_failure():
    """URL-based sources raise SourceLoadError carrying already-loaded docs."""
    source = SourceConfig(
        name="api",
        type="markdown",
        urls=[
            "https://example.com/ok.md",
            "https://example.com/bad.md",
        ],
    )
    ok_response = MagicMock()
    ok_response.text = "# OK"
    ok_response.raise_for_status.return_value = None
    bad_response = MagicMock()
    bad_response.raise_for_status.side_effect = Exception("timeout")

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=[ok_response, bad_response],
    ):
        with pytest.raises(SourceLoadError) as exc_info:
            list(load_documents_from_source(source))

    assert len(exc_info.value.documents) == 1
    assert exc_info.value.documents[0].url == "https://example.com/ok.md"
