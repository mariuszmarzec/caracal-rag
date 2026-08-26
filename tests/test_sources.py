from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from caracal_rag.config import SourceConfig
from caracal_rag.sources import (
    Document,
    _github_list_md_files,
    github_md_doc_dir_documents,
    infer_type,
    load_documents_from_source,
)


def _make_response(json_data, links=None):
    """Create a mock requests.Response."""
    response = MagicMock()
    response.json.return_value = json_data
    response.links = links or {}
    response.raise_for_status.return_value = None
    return response


def test_github_list_md_files_flat_dir():
    """Files in a flat directory are listed correctly."""
    contents = [
        {"type": "file", "name": "login.md", "path": "docs/login.md"},
        {"type": "file", "name": "config.json", "path": "docs/config.json"},
        {"type": "file", "name": "readme.md", "path": "docs/readme.md"},
    ]

    mock_response = _make_response(contents)
    with patch("caracal_rag.sources.requests.get", return_value=mock_response):
        files = list(_github_list_md_files("owner/repo", "docs", "master"))

    assert sorted(files) == ["docs/login.md", "docs/readme.md"]


def test_github_list_md_files_nested_dirs():
    """Sub-directories are traversed recursively."""
    root_contents = [
        {"type": "file", "name": "top.md", "path": "docs/top.md"},
        {"type": "dir", "name": "api", "path": "docs/api"},
        {"type": "dir", "name": "guides", "path": "docs/guides"},
    ]
    api_contents = [
        {"type": "file", "name": "auth.md", "path": "docs/api/auth.md"},
    ]
    guides_contents = [
        {"type": "file", "name": "intro.md", "path": "docs/guides/intro.md"},
    ]

    responses = [
        _make_response(root_contents),
        _make_response(api_contents),
        _make_response(guides_contents),
    ]

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=responses,
    ):
        files = list(_github_list_md_files("owner/repo", "docs", "master"))

    assert sorted(files) == [
        "docs/api/auth.md",
        "docs/guides/intro.md",
        "docs/top.md",
    ]


def test_github_list_md_files_single_file_response():
    """GitHub returns a single dict (not a list) when path is a single file."""
    single = {"type": "file", "name": "only.md", "path": "docs/only.md"}
    mock_response = _make_response(single)

    with patch("caracal_rag.sources.requests.get", return_value=mock_response):
        files = list(_github_list_md_files("owner/repo", "docs/only.md", "master"))

    assert files == ["docs/only.md"]


def test_github_md_doc_dir_documents_fetches_content():
    """Each discovered md file is fetched as raw content and yielded."""
    listing = [
        {"type": "file", "name": "login.md", "path": "docs/api/login.md"},
    ]

    listing_response = _make_response(listing)

    raw_response = MagicMock()
    raw_response.text = "# Login\n\nDetails here."
    raw_response.raise_for_status.return_value = None

    with patch(
        "caracal_rag.sources.requests.get",
        side_effect=[listing_response, raw_response],
    ):
        docs = list(
            github_md_doc_dir_documents(
                source_name="fiteo-docs",
                repo="mariuszmarzec/fiteo",
                path="docs",
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
    assert doc.content == "# Login\n\nDetails here."
    assert doc.content_hash == Document.hash("# Login\n\nDetails here.")


def test_load_documents_from_source_github_md_doc_dir():
    """Config with type github_md_doc_dir delegates to the github fetcher."""
    source = SourceConfig(
        name="fiteo",
        type="github_md_doc_dir",
        repo="mariuszmarzec/fiteo",
        path="docs",
        branch="master",
    )

    listing = [
        {"type": "file", "name": "login.md", "path": "docs/api/login.md"},
    ]
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
    """When branch is None, defaults to 'master'."""
    source = SourceConfig(
        name="fiteo",
        type="github_md_doc_dir",
        repo="mariuszmarzec/fiteo",
        path="docs",
        branch=None,
    )

    listing_response = _make_response([])
    with patch(
        "caracal_rag.sources.requests.get",
        return_value=listing_response,
    ):
        docs = list(load_documents_from_source(source))

    assert docs == []


def test_source_config_requires_repo_and_path():
    """SourceConfig with repo but no path raises."""
    with pytest.raises(ValueError, match="must define repo and path"):
        SourceConfig(name="bad", type="github_md_doc_dir", repo="o/r", path=None)


def test_source_config_requires_urls_or_url_or_repo():
    """SourceConfig with nothing raises."""
    with pytest.raises(ValueError, match="must define urls, url, or repo\\+path"):
        SourceConfig(name="bad", type="markdown", urls=None, url=None)


def test_infer_type_github_paths():
    assert infer_type("docs/api/login.md") == "markdown"
    assert infer_type("docs/api/data.json") == "json"
    assert infer_type("docs/api/config.yaml") == "yaml"
