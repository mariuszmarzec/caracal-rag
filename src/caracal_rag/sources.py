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


def load_documents_from_config(config_path: str) -> Iterable[Document]:
    from caracal_rag.config import AppConfig, SourceConfig
    from caracal_rag.chunking import infer_type

    app_config = AppConfig.from_yaml(config_path)

    for source in app_config.sources:
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
