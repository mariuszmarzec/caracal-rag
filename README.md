# caracal-rag

Caracal RAG - a retrieval-augmented generation toolkit.

## Project structure

```
src/caracal_rag/          package source (src layout)
tests/                    pytest test suite
.github/workflows/ci.yml  CI: lint (ruff) + tests (pytest) on PRs and main
```

## What is caracal-rag?

`caracal-rag` provides a local indexing pipeline that downloads knowledge sources, chunks documents, generates embeddings through a local LiteLLM proxy, and stores vectors in a remote Chroma database. It also exposes an MCP server so an LLM agent can search indexed knowledge.

## Commands

- `python -m caracal_rag check`
- `python -m caracal_rag index`
- `python -m caracal_rag index --source fiteo-api`

## Configuration

Copy `config/sources.example.yaml` into your workspace and configure LiteLLM, Chroma, and source URLs there. Provide secrets via environment variables; do not commit credentials.
