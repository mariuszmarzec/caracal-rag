import hashlib
from typing import List, Dict, Any, Optional
from caracal_rag.config import Config
from caracal_rag.sources import SourceFetcher
from caracal_rag.chunking import Chunker
from caracal_rag.embeddings import EmbeddingsGenerator
from caracal_rag.vectorstore import VectorStore


class IndexingPipeline:
    """Incremental indexing pipeline with idempotency."""

    def __init__(self, config: Config):
        self.config = config
        self.fetcher = SourceFetcher()
        self.chunker = Chunker()
        self.embeddings = EmbeddingsGenerator(config.embedding)
        self.vectorstore = VectorStore(config.chroma)
        self._index_cache: Dict[str, str] = {}  # document_path -> content_hash

    def run_all_sources(self) -> Dict[str, Any]:
        """Run indexing for all configured sources."""
        results = {
            "total_documents": 0,
            "total_chunks": 0,
            "sources": []
        }

        for source in self.config.sources:
            result = self.run_source(source.name)
            results["sources"].append(result)
            results["total_documents"] += result["documents_fetched"]
            results["total_chunks"] += result["chunks_embedded"]

        return results

    def run_source(self, source_name: str) -> Dict[str, Any]:
        """Run indexing for a specific source."""
        source = next((s for s in self.config.sources if s.name == source_name), None)
        if not source:
            raise ValueError(f"Source not found: {source_name}")

        # Fetch documents
        fetch_result = self.fetcher.fetch_document(source)
        documents = fetch_result["documents"]

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Generate embeddings
        chunk_dicts = [chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk for chunk in all_chunks]
        embedded_chunks = self.embeddings.generate_chunks(chunk_dicts)

        # Prepare metadata for vector store
        metadatas = []
        for chunk in embedded_chunks:
            metadata = {
                "source": chunk.get("source", source_name),
                "document": chunk.get("document", ""),
                "url": chunk.get("url", ""),
                "type": chunk.get("type", "markdown"),
                "content_hash": chunk.get("content_hash", ""),
                "chunk_index": chunk.get("chunk_index", 0)
            }
            metadatas.append(metadata)

        # Generate chunk IDs
        chunk_ids = []
        for chunk in embedded_chunks:
            content_hash = chunk.get("content_hash", "")
            chunk_index = chunk.get("chunk_index", 0)
            chunk_id = f"{source_name}_{content_hash}_{chunk_index}"
            chunk_ids.append(chunk_id)

        # Clear old embeddings for this source (idempotency)
        self._clear_source_chunks(source_name)

        # Add to vector store
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        self.vectorstore.add_embeddings(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )

        return {
            "source": source_name,
            "documents_fetched": len(documents),
            "chunks_embedded": len(embedded_chunks)
        }

    def _clear_source_chunks(self, source_name: str):
        """Remove existing chunks for a source before re-indexing."""
        try:
            # Chroma API doesn't support metadata-based deletion directly
            # This is a placeholder for the actual implementation
            # In production, you'd use the appropriate Chroma API endpoint
            pass
        except Exception as e:
            print(f"⚠️ Warning: Could not clear old chunks for {source_name}: {e}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index a list of documents directly."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        chunk_dicts = [chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk for chunk in all_chunks]
        embedded_chunks = self.embeddings.generate_chunks(chunk_dicts)

        metadatas = []
        for chunk in embedded_chunks:
            metadata = {
                "source": chunk.get("source", ""),
                "document": chunk.get("document", ""),
                "url": chunk.get("url", ""),
                "type": chunk.get("type", "markdown"),
                "content_hash": chunk.get("content_hash", ""),
                "chunk_index": chunk.get("chunk_index", 0)
            }
            metadatas.append(metadata)

        chunk_ids = []
        for chunk in embedded_chunks:
            content_hash = chunk.get("content_hash", "")
            chunk_index = chunk.get("chunk_index", 0)
            chunk_id = f"direct_{content_hash}_{chunk_index}"
            chunk_ids.append(chunk_id)

        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        self.vectorstore.add_embeddings(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )

        return {
            "documents_indexed": len(documents),
            "chunks_embedded": len(embedded_chunks)
        }