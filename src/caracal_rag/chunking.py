from typing import List, Dict, Any, Iterable
from dataclasses import dataclass
from caracal_rag.sources import DocumentInfo


@dataclass
class Chunk:
    source: str
    document: str
    url: str
    type: str
    content_hash: str
    chunk_index: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "document": self.document,
            "url": self.url,
            "type": self.type,
            "content_hash": self.content_hash,
            "chunk_index": self.chunk_index,
            "text": self.text
        }


class Chunker:
    """Markdown-aware chunker with metadata generation."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """Split a document into meaningful chunks."""
        chunks = []
        text = document.get("content", "")
        doc_name = document.get("document", "")
        url = document.get("url", "")
        source = document.get("source", "")
        doc_type = document.get("type", "markdown")
        content_hash = document.get("content_hash", "")

        # Split by paragraphs
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    source, doc_name, url, doc_type, content_hash, chunk_index, current_chunk
                ))
                chunk_index += 1

                # Overlap logic
                words = current_chunk.split()
                overlap_text = ' '.join(words[-self.overlap // 10:]) if len(words) > self.overlap // 10 else ""
                current_chunk = overlap_text + '\n\n' + paragraph
            else:
                current_chunk += ('\n\n' if current_chunk else '') + paragraph

        if current_chunk:
            chunks.append(self._create_chunk(
                source, doc_name, url, doc_type, content_hash, chunk_index, current_chunk
            ))

        return chunks

    def _create_chunk(self, source: str, document: str, url: str, type: str,
                     content_hash: str, chunk_index: int, text: str) -> Chunk:
        return Chunk(
            source=source,
            document=document,
            url=url,
            type=type,
            content_hash=content_hash,
            chunk_index=chunk_index,
            text=text
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks