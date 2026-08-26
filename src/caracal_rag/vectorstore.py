import requests
from typing import List, Dict, Any, Optional
from caracal_rag.config import ChromaConfig


class VectorStore:
    """Chroma remote vector-store integration."""

    def __init__(self, config: Optional[ChromaConfig] = None):
        self.config = config or ChromaConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to Chroma API."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def create_collection(self) -> Dict[str, Any]:
        """Create or get the target collection."""
        try:
            return self._make_request("GET", f"/api/v1/collections/{self.config.collection}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return self._make_request("POST", "/api/v1/collections", json={
                    "name": self.config.collection
                })
            raise

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]],
                      ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add embeddings to the vector store."""
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(embeddings))]

        payload = {
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids
        }

        return self._make_request("POST", "/api/v1/embeddings", json=payload)

    def query_similar(self, embedding: List[float], limit: int = 5,
                     where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query for similar embeddings."""
        payload = {
            "query_embeddings": [embedding],
            "n_results": limit,
            "where": where or {}
        }

        return self._make_request("POST", "/api/v1/query", json=payload)

    def delete_collection(self) -> Dict[str, Any]:
        """Delete the collection."""
        return self._make_request("DELETE", f"/api/v1/collections/{self.config.collection}")

    def clear_collection(self) -> Dict[str, Any]:
        """Remove all embeddings from the collection."""
        return self._make_request("POST", "/api/v1/reset")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return self._make_request("GET", f"/api/v1/collections/{self.config.collection}")


class ChromaClient:
    """Helper class for Chroma configuration and utilities."""

    def __init__(self, host: str = "localhost", port: int = 3400, collection: str = "caracal-base"):
        self.config = ChromaConfig(host=host, port=port, collection=collection)

    def get_connection_string(self) -> str:
        """Get Chroma connection string."""
        return f"http://{self.config.host}:{self.config.port}"

    def get_health_check_url(self) -> str:
        """Get health check endpoint URL."""
        return f"{self.get_connection_string()}/api/v1/collections"