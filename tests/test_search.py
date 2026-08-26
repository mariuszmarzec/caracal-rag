"""Search-related tests for caracal_rag."""

import pytest
from unittest.mock import Mock, patch
from caracal_rag.vectorstore import VectorStore


def test_query_similar_basic():
    """Test basic query functionality."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test"}, {"source": "test"}]],
            "distances": [[0.1, 0.2]]
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test"

        store = VectorStore(config)
        result = store.query_similar([0.1, 0.2, 0.3], limit=5)

        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result
        mock_request.assert_called_once()


def test_query_similar_with_where():
    """Test query with where clause."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]]
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test"

        store = VectorStore(config)
        where_clause = {"source": "test-source"}
        result = store.query_similar([0.1, 0.2], limit=3, where=where_clause)

        # Verify the where clause was passed
        call_args = mock_request.call_args
        assert call_args[1]["json"]["where"] == where_clause


def test_get_collection_info():
    """Test getting collection information."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "name": "test-collection",
            "metadata": {"description": "Test collection"}
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test-collection"

        store = VectorStore(config)
        result = store.get_collection_info()

        assert "name" in result
        assert "metadata" in result
        mock_request.assert_called_with("GET", "/api/v1/collections/test-collection")


def test_clear_collection():
    """Test clearing the collection."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {"success": True}

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test"

        store = VectorStore(config)
        result = store.clear_collection()

        assert result["success"] is True
        mock_request.assert_called_with("POST", "/api/v1/reset")


def test_delete_collection():
    """Test deleting the collection."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {"success": True}

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test-collection"

        store = VectorStore(config)
        result = store.delete_collection()

        assert result["success"] is True
        mock_request.assert_called_with("DELETE", "/api/v1/collections/test-collection")


def test_create_collection_existing():
    """Test creating collection when it already exists."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "name": "existing-collection",
            "metadata": {}
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "existing-collection"

        store = VectorStore(config)
        result = store.create_collection()

        assert "name" in result
        # Should have made a GET request first
        assert mock_request.call_count >= 1


def test_add_embeddings():
    """Test adding embeddings to vector store."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "ids": ["id1", "id2"],
            "metadatas": [{"source": "test"}],
            "embeddings": [[0.1, 0.2]]
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test"

        store = VectorStore(config)
        result = store.add_embeddings(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"source": "test"}, {"source": "test2"}],
            ids=["custom_id1", "custom_id2"]
        )

        assert "ids" in result
        assert "metadatas" in result
        assert "embeddings" in result


def test_add_embeddings_auto_ids():
    """Test adding embeddings with auto-generated IDs."""
    with patch.object(VectorStore, '_make_request') as mock_request:
        mock_request.return_value = {
            "ids": ["chunk_0", "chunk_1"],
            "metadatas": [{"source": "test"}],
            "embeddings": [[0.1, 0.2]]
        }

        config = Mock()
        config.host = "localhost"
        config.port = 8000
        config.collection = "test"

        store = VectorStore(config)
        result = store.add_embeddings(
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )

        assert len(result["ids"]) == 2
        assert all(id.startswith("chunk_") for id in result["ids"])