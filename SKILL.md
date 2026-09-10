# Caracal RAG - Available Skills

This document describes all the skills and capabilities available in the Caracal RAG toolkit.

## Core Skills (MCP Tools)

The Caracal RAG MCP server exposes a single tool that agents can use:

### `search_api`

**Description:** Semantic search over indexed documentation chunks.

**Parameters:**
- `query` (required): Search query string
- `top_k` (optional): Number of results to return (default: 5)

**Returns:** List of relevant document chunks with source metadata including:
- Source name
- Document name
- URL (path to source file)
- Content (relevant text chunk)

**Example usage:**
```
search_api(query="How to configure LiteLLM", top_k=5)
```

## CLI Commands

### `check`
Validates configuration and connectivity to LiteLLM and Chroma without modifying the database.

```bash
python -m caracal_rag check
```

### `index`
Index documents from configured sources. Can index a specific source or all sources.

```bash
# Index all sources
python -m caracal_rag index

# Index a specific source
python -m caracal_rag index --source example_knowledge_funny_jokes
```

### `mcp`
Start the MCP server for agent integration.

```bash
python -m caracal_rag mcp
```

## Source Types

Caracal RAG supports three source types for indexing knowledge:

### 1. Local Markdown Files
Index all `.md` files from a local directory recursively.

```yaml
sources:
  - name: example_knowledge_funny_jokes
    type: local
    path: ./example_knowledge_funny_jokes
```

### 2. GitHub Markdown Directory
Recursively fetch all `.md` files from a GitHub repository directory.

```yaml
sources:
  - name: fiteo-docs
    type: github_md_doc_dir
    repo: mariuszmarzec/fiteo
    path: docs
    branch: master
```

### 3. URL-based Documents
Index individual markdown files from URLs.

```yaml
sources:
  - name: some-docs
    type: markdown
    url: https://example.com/docs/guide.md
```

## Configuration

Configuration is managed via YAML files (default: `config/sources.example.yaml`):

```yaml
embedding:
  model: text-embedding-3-small  # Embedding model name
  api_base: http://localhost:4000  # LiteLLM endpoint
  api_key: null  # Optional API key

chroma:
  host: localhost  # ChromaDB host
  port: 3400  # ChromaDB port
  collection: caracal-base  # Collection name
  ssl: false  # Enable SSL

sources:
  # ... source configurations
```

Environment variables can also be used:
- `CARACAL_EMBEDDING_MODEL`
- `CARACAL_EMBEDDING_API_BASE`
- `CARACAL_EMBEDDING_API_KEY`
- `CARACAL_CHROMA_HOST`
- `CARACAL_CHROMA_PORT`
- `CARACAL_CHROMA_COLLECTION`
- `CARACAL_CHROMA_SSL`

## Chunking Strategy

Documents are chunked using a smart markdown-aware strategy:
- Splits markdown by heading sections when possible
- Respects `max_chars` limit (default: 1800 characters)
- Falls back to character-based chunking for non-markdown content

## Example Knowledge Sources

The repository includes example knowledge sources in `example_knowledge_funny_jokes/`:
- `magic-words.md`
- `international-swearing-competition.md`
- `politics.md`
- `the-echo.md`
- `the-gravestone.md`
- `the-teachers-signature.md`
- `the-teddy-bear.md`
- `the-worlds-most-modern-bank.md`
