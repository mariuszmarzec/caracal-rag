import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from caracal_rag.config import SourceConfig


class DocumentInfo:
    """Document information from source fetching."""

    def __init__(self, source: str, document: str, url: str, type: str,
                 content: str, content_hash: str, path: str):
        self.source = source
        self.document = document
        self.url = url
        self.type = type
        self.content = content
        self.content_hash = content_hash
        self.path = path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "document": self.document,
            "url": self.url,
            "type": self.type,
            "content": self.content,
            "content_hash": self.content_hash,
            "path": self.path
        }


class SourceFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; caracal-rag/1.0)"
        })

    def fetch_document(self, source: SourceConfig) -> Dict[str, Any]:
        """Fetch documents from a configured source."""
        if source.type == "markdown":
            return self._fetch_markdown_urls(source)
        elif source.type == "github_md_doc_dir":
            return self._fetch_github_dir(source)
        elif source.type == "local_path":
            return self._fetch_local_path(source)
        else:
            raise ValueError(f"Unsupported source type: {source.type}")

    def _fetch_markdown_urls(self, source: SourceConfig) -> Dict[str, Any]:
        """Fetch individual markdown files by raw URL."""
        documents = []

        for url in source.urls:
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                content = response.text
                doc_name = Path(url).stem

                doc_info = DocumentInfo(
                    source=source.name,
                    document=doc_name,
                    url=url,
                    type="markdown",
                    content=content,
                    content_hash=self._calculate_hash(content),
                    path=doc_name
                )

                documents.append(doc_info.to_dict())

            except Exception as e:
                print(f"❌ Failed to fetch {url}: {e}")

        return {
            "source": source.name,
            "documents": documents,
            "total_fetched": len(documents)
        }

    def _fetch_github_dir(self, source: SourceConfig) -> Dict[str, Any]:
        """Recursively fetch all .md files from a GitHub repository directory."""
        repo = source.repo
        path = source.path or ""
        branch = source.branch or "main"

        api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

        try:
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()

            contents = response.json()
            documents = []

            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".md"):
                    download_url = item["download_url"]

                    doc_response = self.session.get(download_url, timeout=30)
                    doc_response.raise_for_status()

                    content = doc_response.text
                    doc_name = item["name"]

                    doc_info = DocumentInfo(
                        source=source.name,
                        document=doc_name,
                        url=download_url,
                        type="github_md_doc_dir",
                        content=content,
                        content_hash=self._calculate_hash(content),
                        path=f"{path}/{doc_name}"
                    )

                    documents.append(doc_info.to_dict())

            return {
                "source": source.name,
                "documents": documents,
                "total_fetched": len(documents)
            }

        except Exception as e:
            print(f"❌ Failed to fetch GitHub directory {api_url}: {e}")
            raise

    def _fetch_local_path(self, source: SourceConfig) -> Dict[str, Any]:
        """Recursively fetch all .md files from a local directory."""
        local_path = Path(source.path)
        documents = []

        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        if not local_path.is_dir():
            raise ValueError(f"Local path is not a directory: {local_path}")

        for md_file in sorted(local_path.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                doc_name = md_file.name
                relative_path = str(md_file.relative_to(local_path))

                doc_info = DocumentInfo(
                    source=source.name,
                    document=doc_name,
                    url=f"file://{md_file.resolve()}",
                    type="local_path",
                    content=content,
                    content_hash=self._calculate_hash(content),
                    path=relative_path
                )

                documents.append(doc_info.to_dict())

            except Exception as e:
                print(f"❌ Failed to read {md_file}: {e}")

        return {
            "source": source.name,
            "documents": documents,
            "total_fetched": len(documents)
        }

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def fetch_all_documents(self, sources: List[SourceConfig]) -> List[Dict[str, Any]]:
        """Fetch all documents from all configured sources."""
        all_documents = []

        for source in sources:
            try:
                result = self.fetch_document(source)
                all_documents.extend(result["documents"])
            except Exception as e:
                print(f"❌ Failed to fetch source {source.name}: {e}")

        return all_documents


class GitHubDocDir:
    """Helper for GitHub markdown directory source configuration."""

    def __init__(self, repo: str, path: str = "docs", branch: str = "main"):
        self.repo = repo
        self.path = path
        self.branch = branch

    def to_source_config(self, name: str) -> SourceConfig:
        return SourceConfig(
            name=name,
            type="github_md_doc_dir",
            repo=self.repo,
            path=self.path,
            branch=self.branch
        )