# caracal-rag 🐈

Kotlin RAG example service for the fiteo backend's OpenAPI-generated documentation.

This repository is the research-backed proposal for **card R-1 / R-2** of [fiteo issue #6 — "RAG for backend"](https://github.com/mariuszmarzec/fiteo/issues/6): a small retrieval service over `docs/api/*.md` (one markdown document per endpoint) that answers `POST /search` queries with the top-k relevant documents. It replaces the earlier plan's Python (FastAPI + chromadb) design with a **Kotlin/JVM** stack.

Status: **proposal** — research + architecture only, no implementation yet. Open questions are at the bottom; implementation starts once they're answered.

## Why Kotlin? (the "why Python?" answer)

The earlier plan chose Python (FastAPI + chromadb) as the *simplest demo stack*, not because Python was required:

- FastAPI is a ~10-line way to expose `POST /search`, and chromadb is a Python-native vector server — so the demo had almost zero glue code.
- Most public RAG examples are Python-first, so it was also the path of least resistance for copy-paste learning.

That convenience comes with real costs for this project:

- **A second runtime and second language** in a stack that is otherwise all-Kotlin (the fiteo backend is Ktor/Kotlin). Two languages means two toolchains, two sets of idioms, and more context to maintain — a lot for a small example service.
- **No shared code with fiteo**: types, config, and developer workflows (Gradle, detekt, ktlint-style conventions) would not carry over.

**Kotlin is viable — there is a ready-to-use ecosystem.** Details in the next section. The honest caveats: LangChain4j's vector-store modules currently ship on a beta line (see table), and there is less "blog-post parity" example material than Python. Neither blocks this use case (small corpus, one embedding model, one endpoint).

## Research: ready-to-use JVM/Kotlin RAG options

### Option A — LangChain4j (recommended)

[LangChain4j](https://github.com/langchain4j/langchain4j) is the de-facto standard Java/Kotlin LLM framework: a unified API for chat/embedding models and embedding stores, plus a full RAG pipeline (`EmbeddingStoreContentRetriever` + `RetrievalAugmentor`) — see the [RAG tutorial](https://docs.langchain4j.dev/tutorials/rag). It is a plain Java library, so it is called directly from Kotlin.

Current versions (Maven Central `maven-metadata.xml`, checked 2026-08-10):

| Artifact | Version | Status |
|---|---|---|
| `dev.langchain4j:langchain4j` | 1.18.1 | stable |
| `dev.langchain4j:langchain4j-ollama` | 1.18.1 | stable |
| `dev.langchain4j:langchain4j-chroma` | 1.18.1-beta28 | beta line |
| `dev.langchain4j:langchain4j-qdrant` | 1.18.1-beta28 | beta line |
| `dev.langchain4j:langchain4j-pgvector` | 1.18.1-beta28 | beta line |

What is genuinely usable in 2026: embeddings via the [Ollama integration](https://docs.langchain4j.dev/integrations/language-models/ollama) (`OllamaEmbeddingModel`), store + top-k retrieve with scores via the embedding-store modules, and the RAG abstractions on top. The beta-line caveat for vector-store modules is real but acceptable for an example service; the modules are released in lockstep with the core, and the core itself is stable (1.18.1).

### Option B — Spring AI (alternative)

[Spring AI](https://spring.io/projects/spring-ai) 2.0.0 GA (June 2026) is the Spring-ecosystem answer: `spring-ai-starter-model-ollama` plus vector-store starters (Chroma, Qdrant, PGVector), an ETL pipeline, and RAG advisors (see the [reference docs](https://docs.spring.io/spring-ai/reference/index.html)). It is very complete and well documented, but it pulls in Spring Boot — heavier than needed for a one-endpoint example, and less consistent with fiteo's Ktor style.

### Option C — Hand-rolled Ktor (minimal)

Ollama's API is plain HTTP, so nothing forces a framework: a Ktor service can `POST http://localhost:11434/api/embed` directly ([Ollama API: Generate embeddings](https://docs.ollama.com/api/embed)) and do similarity search in-process — plain cosine similarity over a small corpus, or HNSW via [JVector](https://github.com/jbellis/jvector) (the Apache-licensed library used by Cassandra/Elastic). Maximum control, minimum dependencies; you reimplement what LangChain4j already provides.

## Local embeddings via Ollama

Ollama is a language-agnostic HTTP server — the Python plan's choice of `nomic-embed-text` works unchanged from Kotlin:

```bash
curl http://localhost:11434/api/embed -d '{"model": "nomic-embed-text", "input": "search query"}'
# → {"embeddings": [[...]], ...}
```

LangChain4j's `OllamaEmbeddingModel` wraps exactly this call. No Python runtime is needed anywhere in the pipeline.

## Vector stores with JVM clients

| Store | JVM access | Notes |
|---|---|---|
| Chroma | `langchain4j-chroma` (bundled client) | parity with the Python plan's store |
| Qdrant | `langchain4j-qdrant`, or the official [`io.qdrant:client`](https://github.com/qdrant/java-client) (1.14.x) | purpose-built vector DB, easy Docker |
| PostgreSQL + pgvector | `langchain4j-pgvector` (JDBC) | if fiteo already runs Postgres — no extra service |
| In-memory | LangChain4j `InMemoryEmbeddingStore`, or JVector | zero infrastructure, fine for a demo corpus |

## Proposed Kotlin architecture

Ktor service + LangChain4j, mirroring the Python plan's R-1/R-2 acceptance criteria:

```
docs/api/*.md  (from fiteo)
      │  ./gradlew index    (ingest)
      ▼
split per endpoint → OllamaEmbeddingModel (nomic-embed-text) → Chroma/Qdrant
                                                                │
POST /search {query, topK}  ← ./gradlew serve (Ktor) ←── top-k via EmbeddingStoreContentRetriever
```

- **Stack**: Kotlin JVM, Gradle (Kotlin DSL), Ktor server, kotlinx-serialization, LangChain4j 1.18.1 (`langchain4j-ollama` + one store module).
- **Gradle tasks** (the `make index` / `make serve` equivalents): `./gradlew index` ingests and embeds `docs/api/*.md`; `./gradlew serve` runs the Ktor service with `POST /search`.
- **Acceptance (from the Python plan)**: a golden query set mapping each query to its expected endpoint doc(s); **recall@3** must return the expected doc in the top-3; encoded as tests and runnable via `./gradlew check`.
- **Response**: ranked list of `{endpoint, doc path, score}` — one document per fiteo endpoint, so retrieval answers "which endpoint(s) handle this user intent".

## Open questions (for Mariusz)

1. **Repo visibility** — created public (matching fiteo); keep it public?
2. **Vector store** — Chroma (parity with the Python plan), Qdrant, pgvector, or in-memory first?
3. **Scaffold now or after answers?** This PR is docs-only; a follow-up can add the Gradle scaffold + implementation.
4. **Ollama assumptions** — assume Ollama on `localhost:11434` with `nomic-embed-text` pulled (as in the Python plan)? Any GPU/CPU constraints?
5. **Doc sourcing** — pull fiteo's `docs/api/*.md` via git submodule, tarball download, or a copy script?

## Sources

- LangChain4j: https://github.com/langchain4j/langchain4j · docs: https://docs.langchain4j.dev/ · RAG tutorial: https://docs.langchain4j.dev/tutorials/rag
- LangChain4j Ollama integration: https://docs.langchain4j.dev/integrations/language-models/ollama
- Versions: https://repo1.maven.org/maven2/dev/langchain4j/ (`maven-metadata.xml`, checked 2026-08-10)
- Spring AI: https://spring.io/projects/spring-ai · reference: https://docs.spring.io/spring-ai/reference/index.html
- Ollama embeddings API: https://docs.ollama.com/api/embed
- Qdrant Java client: https://github.com/qdrant/java-client
- JVector: https://github.com/jbellis/jvector
- fiteo issue #6: https://github.com/mariuszmarzec/fiteo/issues/6
