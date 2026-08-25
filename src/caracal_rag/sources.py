from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


@dataclass(frozen=True)
class Document:
    source: str
    name: str
    url: str
    type: str
    content: str
    content_hash: str

    @staticmethod
    def hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def fetch_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def infer_type(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".md"):
        return "markdown"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return "yaml"
    return "text"


def _github_list_md_files(repo: str, path: str, branch: str) -> Iterable[str]:
    """List .md files recursively in a GitHub directory.

    Uses the GitHub contents API (paginated) with public (unauthenticated)
    access. Yields relative file paths.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    while url:
        response = requests.get(url, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        entries = response.json()
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if entry.get("type") == "dir":
                yield from _github_list_md_files(
                    repo, entry["path"], branch
                )
            elif entry.get("type") == "file":
                if entry["name"].lower().endswith(".md"):
                    yield entry["path"]
        # Pagination
        next_url = response.links.get("next", {}).get("url")
        url = next_url


def github_md_doc_dir_documents(
    source_name: str,
    repo: str,
    path: str,
    branch: str = "master",
) -> Iterable[Document]:
    """Fetch all markdown files from a GitHub directory recursively."""
    owner_repo = repo
    for file_path in _github_list_md_files(owner_repo, path, branch):
        url = (
            f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{file_path}"
        )
        try:
            content = fetch_text(url)
        except Exception:
            continue
        name = Path(file_path).name or url
        doc_type = infer_type(name)
        yield Document(
            source=source_name,
            name=name,
            url=url,
            type=doc_type,
            content=content,
            content_hash=Document.hash(content),
        )


def load_documents_from_source(source) -> Iterable[Document]:
    """Load documents from a single source config based on its type."""
    if source.type == "github_md_doc_dir":
        if not source.repo or not source.path:
            raise ValueError(
                f"Source {source.name!r} of type github_md_doc_dir must define repo and path"
            )
        branch = source.branch or "master"
        yield from github_md_doc_dir_documents(
            source_name=source.name,
            repo=source.repo,
            path=source.path,
            branch=branch,
        )
        return

    urls = source.urls or []
    if source.url:
        urls.append(source.url)
    for url in urls:
        try:
            text = fetch_text(url)
        except Exception:
            continue
        name = Path(url).name or url
        doc_type = infer_type(name)
        yield Document(
            source=source.name,
            name=name,
            url=url,
            type=doc_type,
            content=text,
            content_hash=Document.hash(text),
        )


def load_documents_from_config(config_path: str) -> Iterable[Document]:
    from caracal_rag.config import AppConfig

    app_config = AppConfig.from_yaml(config_path)

    for source in app_config.sources:
        yield from load_documents_from_source(source)
