"""Source-related tests for caracal_rag."""

import pytest
from unittest.mock import Mock, patch
from caracal_rag.sources import SourceFetcher, DocumentInfo
from caracal_rag.config import SourceConfig


def test_source_fetcher_markdown_urls():
    """Test fetching markdown documents by URL."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    with patch('caracal_rag.sources.requests.Session') as mock_session:
        mock_get = Mock(return_value=mock_response)
        mock_session.return_value.get = mock_get

        source = SourceConfig(
            name="test-source",
            type="markdown",
            urls=["https://example.com/doc.md"]
        )

        fetcher = SourceFetcher()
        result = fetcher.fetch_document(source)

        assert result["source"] == "test-source"
        assert len(result["documents"]) == 1
        assert result["documents"][0]["source"] == "test-source"
        assert result["documents"][0]["content"] == "Test content"


def test_source_fetcher_github_dir():
    """Test fetching documents from GitHub directory."""
    mock_api_response = Mock()
    mock_api_response.json.return_value = [
        {
            "type": "file",
            "name": "doc1.md",
            "download_url": "https://github.com/user/repo/raw/main/docs/doc1.md"
        },
        {
            "type": "file",
            "name": "doc2.md",
            "download_url": "https://github.com/user/repo/raw/main/docs/doc2.md"
        }
    ]
    mock_api_response.raise_for_status.return_value = None

    mock_download_response = Mock()
    mock_download_response.text = "Document content"
    mock_download_response.raise_for_status.return_value = None

    with patch('caracal_rag.sources.requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        # Mock API call for directory listing
        mock_session_instance.get.side_effect = [
            mock_api_response,  # API call
            mock_download_response,  # doc1 download
            mock_download_response  # doc2 download
        ]

        source = SourceConfig(
            name="github-source",
            type="github_md_doc_dir",
            repo="user/repo",
            path="docs",
            branch="main"
        )

        fetcher = SourceFetcher()
        result = fetcher.fetch_document(source)

        assert result["source"] == "github-source"
        assert len(result["documents"]) == 2
        assert all(doc["source"] == "github-source" for doc in result["documents"])


def test_source_fetcher_unsupported_type():
    """Test that unsupported source type raises ValueError."""
    source = SourceConfig(
        name="invalid-source",
        type="invalid-type",
        urls=[]
    )

    fetcher = SourceFetcher()

    with pytest.raises(ValueError, match="Unsupported source type: invalid-type"):
        fetcher.fetch_document(source)


def test_document_info():
    """Test DocumentInfo class."""
    doc = DocumentInfo(
        source="test",
        document="doc.md",
        url="https://example.com/doc.md",
        type="markdown",
        content="Test content",
        content_hash="abc123",
        path="doc.md"
    )

    assert doc.source == "test"
    assert doc.document == "doc.md"
    assert doc.url == "https://example.com/doc.md"
    assert doc.type == "markdown"
    assert doc.content == "Test content"
    assert doc.content_hash == "abc123"
    assert doc.path == "doc.md"

    doc_dict = doc.to_dict()
    assert doc_dict["source"] == "test"
    assert doc_dict["document"] == "doc.md"
    assert "content" in doc_dict


def test_github_docdir_helper():
    """Test GitHubDocDir helper."""
    from caracal_rag.sources import GitHubDocDir

    helper = GitHubDocDir("myorg/myrepo", "docs/api", "main")
    source_config = helper.to_source_config("my-docs")

    assert isinstance(source_config, SourceConfig)
    assert source_config.name == "my-docs"
    assert source_config.type == "github_md_doc_dir"
    assert source_config.repo == "myorg/myrepo"
    assert source_config.path == "docs/api"
    assert source_config.branch == "main"


def test_source_fetcher_fetch_all_documents():
    """Test fetching documents from multiple sources."""
    source1 = SourceConfig(
        name="source1",
        type="markdown",
        urls=["https://example.com/doc1.md"]
    )
    source2 = SourceConfig(
        name="source2",
        type="markdown",
        urls=["https://example.com/doc2.md"]
    )

    mock_response1 = Mock()
    mock_response1.text = "Content 1"
    mock_response1.raise_for_status.return_value = None

    mock_response2 = Mock()
    mock_response2.text = "Content 2"
    mock_response2.raise_for_status.return_value = None

    with patch('caracal_rag.sources.requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        mock_session_instance.get.side_effect = [mock_response1, mock_response2]

        fetcher = SourceFetcher()
        result = fetcher.fetch_all_documents([source1, source2])

        assert len(result) == 2
        assert result[0]["content"] == "Content 1"
        assert result[1]["content"] == "Content 2"
        assert result[0]["source"] == "source1"
        assert result[1]["source"] == "source2"