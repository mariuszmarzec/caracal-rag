"""Core functionality tests for caracal_rag."""

import pytest
import yaml
from pathlib import Path
from caracal_rag.config import Config, EmbeddingConfig, ChromaConfig, SourceConfig
from caracal_rag.sources import SourceFetcher
from caracal_rag.chunking import Chunker, Chunk
from caracal_rag.embeddings import EmbeddingsGenerator
from caracal_rag.vectorstore import VectorStore


def test_config_load_from_yaml():
    """Test configuration loading from YAML file."""
    config_data = {
        "embedding": {
            "model": "text-embedding-3-small",
            "api_base": "http://localhost:4000"
        },
        "chroma": {
            "host": "localhost",
            "port": 8000,
            "collection": "test-collection"
        },
        "sources": [
            {
                "name": "test-source",
                "type": "markdown",
                "urls": ["https://example.com/doc.md"]
            }
        ]
    }

    config_path = Path("test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    config = Config.load(str(config_path))
    assert config.embedding.model == "text-embedding-3-small"
    assert config.chroma.host == "localhost"
    assert config.chroma.port == 8000
    assert config.chroma.collection == "test-collection"
    assert len(config.sources) == 1
    assert config.sources[0].name == "test-source"

    config_path.unlink()


def test_source_config_creation():
    """Test SourceConfig dataclass."""
    source = SourceConfig(
        name="test",
        type="markdown",
        urls=["https://example.com/doc.md"]
    )
    assert source.name == "test"
    assert source.type == "markdown"
    assert len(source.urls) == 1


def test_embed_config_defaults():
    """Test EmbeddingConfig default values."""
    config = EmbeddingConfig()
    assert config.model == "text-embedding-3-small"
    assert config.api_base == "http://localhost:3001/v1/embeddings"


def test_chroma_config_defaults():
    """Test ChromaConfig default values."""
    config = ChromaConfig()
    assert config.host == "localhost"
    assert config.port == 3400
    assert config.collection == "caracal-base"


def test_chunker_basic():
    """Test basic chunker functionality."""
    chunker = Chunker(chunk_size=100, overlap=20)

    document = {
        "source": "test-source",
        "document": "test.md",
        "url": "https://example.com/test.md",
        "type": "markdown",
        "content": "This is a test document with some content. " * 10,
        "content_hash": "abc123"
    }

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.source == "test-source" for chunk in chunks)
    assert all(chunk.document == "test.md" for chunk in chunks)
    assert all(hasattr(chunk, 'to_dict') for chunk in chunks)


def test_chunk_to_dict():
    """Test Chunk.to_dict method."""
    chunk = Chunk(
        source="test-source",
        document="test.md",
        url="https://example.com/test.md",
        type="markdown",
        content_hash="abc123",
        chunk_index=0,
        text="Test content"
    )

    chunk_dict = chunk.to_dict()
    assert chunk_dict["source"] == "test-source"
    assert chunk_dict["document"] == "test.md"
    assert chunk_dict["chunk_index"] == 0
    assert chunk_dict["text"] == "Test content"


def test_embeddings_generator_initialization():
    """Test EmbeddingsGenerator initialization with config."""
    embedding_config = EmbeddingConfig(model="text-embedding-test")
    generator = EmbeddingsGenerator(embedding_config)

    assert generator.config.model == "text-embedding-test"


def test_embeddings_get_model_name():
    """Test getting model name from embeddings generator."""
    generator = EmbeddingsGenerator()
    assert generator.get_model_name() == "text-embedding-3-small"


def test_vectorstore_initialization():
    """Test VectorStore initialization."""
    config = ChromaConfig(host="localhost", port=8000, collection="test")
    store = VectorStore(config)

    assert store.config.host == "localhost"
    assert store.config.port == 8000
    assert store.config.collection == "test"
    assert store.base_url == "http://localhost:8000"


def test_github_docdir_helper():
    """Test GitHubDocDir helper class."""
    from caracal_rag.sources import GitHubDocDir

    helper = GitHubDocDir("mariuszmarzec/fiteo", "docs/api", "master")
    source_config = helper.to_source_config("test-source")

    assert source_config.name == "test-source"
    assert source_config.type == "github_md_doc_dir"
    assert source_config.repo == "mariuszmarzec/fiteo"
    assert source_config.path == "docs/api"
    assert source_config.branch == "master"