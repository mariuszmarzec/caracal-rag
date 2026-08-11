# caracal-rag

Caracal RAG - a retrieval-augmented generation toolkit (project skeleton).

## Project structure

```
src/caracal_rag/          package source (src layout)
tests/                    pytest test suite
.github/workflows/ci.yml  CI: lint (ruff) + tests (pytest) on PRs and main
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
```
