from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
sys.modules["chromadb"] = chromadb_stub

litellm_stub = types.ModuleType("litellm")
litellm_stub.embedding = lambda *args, **kwargs: {"data": [{"embedding": [0.1, 0.2]}]}
litellm_stub.api_base = None
litellm_stub.api_key = None
sys.modules["litellm"] = litellm_stub

from caracal_rag import benchmark  # noqa: E402
from caracal_rag.manifest import build_manifest, write_manifest  # noqa: E402


def _config():
    return SimpleNamespace(
        embedding=SimpleNamespace(
            model="bge-m3",
            api_base="http://localhost:3001/v1",
            api_key="test-key",
        ),
        chroma=SimpleNamespace(
            host="localhost",
            port=3400,
            ssl=False,
            collection="test-collection",
        ),
    )


def test_audit_index_reports_chunk_health_without_embedding_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    collection = SimpleNamespace(
        get=lambda include=None: {
            "ids": ["c1", "c2", "c3", "staged"],
            "documents": ["duplicate chunk", "duplicate chunk", "  ", "not committed"],
            "metadatas": [
                {
                    "source": "api-docs",
                    "document": "auth.md",
                    "url": "https://example.com/auth.md",
                    "embedding_model": "bge-m3",
                    "embedding_dim": 2,
                    "chunk_index": 0,
                    "caracal_staging": False,
                },
                {
                    "source": "api-docs",
                    "document": "auth.md",
                    "url": "https://example.com/auth.md",
                    "embedding_model": "bge-m3",
                    "embedding_dim": 2,
                    "chunk_index": 1,
                    "caracal_staging": False,
                },
                {
                    "source": "other",
                    "document": "empty.md",
                    "url": "https://example.com/empty.md",
                    "embedding_dim": None,
                    "caracal_staging": False,
                },
                {
                    "source": "api-docs",
                    "document": "staged.md",
                    "url": "https://example.com/staged.md",
                    "embedding_model": "bge-m3",
                    "embedding_dim": 2,
                    "caracal_staging": True,
                },
            ],
        }
    )
    monkeypatch.setattr(benchmark, "_collection_for", lambda config: collection)
    monkeypatch.setenv("CARACAL_HOME", str(tmp_path))
    write_manifest(
        tmp_path / "index" / "manifest.json",
        build_manifest(_config(), document_count=5, embedding_dim=1024),
    )
    monkeypatch.setattr(
        benchmark,
        "embed_texts",
        lambda *args, **kwargs: pytest.fail("index audit must not call the embedding API"),
    )

    report = benchmark.audit_index(_config())

    assert report["counts"]["total_chunks"] == 4
    assert report["counts"]["committed_chunks"] == 3
    assert report["counts"]["staging_chunks"] == 1
    assert report["counts"]["documents"] == 2
    assert report["counts"]["empty_chunks"] == 1
    assert report["counts"]["duplicate_chunk_text_groups"] == 1
    assert report["counts"]["duplicate_chunk_text_rows"] == 1
    assert report["counts"]["missing_embedding_model_metadata"] == 1
    assert report["embedding_models"] == {"<missing>": 1, "bge-m3": 2}
    assert any("staging chunk" in warning for warning in report["warnings"])
    assert any("manifest embedding dimension" in warning for warning in report["warnings"])
    assert any("manifest document count" in warning for warning in report["warnings"])


def test_load_benchmark_dataset_requires_ground_truth(tmp_path: Path) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps({"version": 1, "cases": [{"id": "q1", "query": "How does auth work?"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="needs at least one"):
        benchmark.load_benchmark_dataset(dataset_path)


def test_retrieval_benchmark_separates_document_hits_from_chunk_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "auth",
                        "query": "How is authentication configured?",
                        "relevant_urls": ["https://example.com/auth.md"],
                        "expected_terms": ["bearer", "access token"],
                        "relevant_chunk_terms": [["bearer", "access token"]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    collection = object()
    monkeypatch.setattr(benchmark, "_collection_for", lambda config: collection)
    monkeypatch.setattr(benchmark, "assert_manifest_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "assert_embedding_model_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "assert_embedding_dim_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "embed_texts", lambda *args, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr(
        benchmark,
        "search_chunks",
        lambda _collection, _vector, top_k: {
            "ids": [["wrong-url", "auth-without-evidence", "auth-evidence"]],
            "documents": [[
                "Bearer access token overview",
                "Authentication endpoint details only",
                "The Authorization header uses Bearer access token.",
            ]],
            "metadatas": [[
                {
                    "source": "api-docs",
                    "document": "overview.md",
                    "url": "https://example.com/overview.md",
                    "chunk_index": 0,
                },
                {
                    "source": "api-docs",
                    "document": "auth.md",
                    "url": "https://example.com/auth.md",
                    "chunk_index": 0,
                },
                {
                    "source": "api-docs",
                    "document": "auth.md",
                    "url": "https://example.com/auth.md",
                    "chunk_index": 1,
                },
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        },
    )

    report = benchmark.benchmark_retrieval(_config(), dataset_path, top_k=5)

    assert report["schema_version"] == 2
    assert report["metrics"]["document_hit_rate_at_1"] == 0.0
    assert report["metrics"]["document_hit_rate_at_5"] == 1.0
    assert report["metrics"]["document_mrr_at_5"] == 0.5
    assert report["metrics"]["evidence_hit_rate_at_1"] == 0.0
    assert report["metrics"]["evidence_hit_rate_at_5"] == 1.0
    assert report["metrics"]["evidence_mrr_at_5"] == round(1 / 3, 4)
    assert report["metrics"]["expected_term_coverage_at_5"] == 1.0
    assert report["expected_term_coverage"] == 1.0

    case = report["cases"][0]
    assert case["document_hit"] is True
    assert case["evidence_hit"] is True
    assert case["evidence_rank"] == 3
    assert case["found_expected_terms_in_target_chunks"] == ["bearer", "access token"]
    assert case["retrieved"][0]["matches_expected_target"] is False
    assert case["retrieved"][0]["found_expected_terms"] == ["bearer", "access token"]
    assert case["retrieved"][0]["is_evidence_chunk"] is False
    assert case["retrieved"][1]["is_evidence_chunk"] is False
    assert case["retrieved"][2]["is_evidence_chunk"] is True
    assert "Bearer" not in case["retrieved"][2]["content_preview"]
    assert len(case["retrieved"]) == 3


def test_top_k_twenty_identifies_evidence_chunk_after_rank_ten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "auth-deep-result",
                        "query": "Where is bearer token setup documented?",
                        "relevant_urls": ["https://example.com/auth.md"],
                        "expected_terms": ["bearer", "access token"],
                        "relevant_chunk_terms": [["bearer", "access token"]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    collection = object()
    monkeypatch.setattr(benchmark, "_collection_for", lambda config: collection)
    monkeypatch.setattr(benchmark, "assert_manifest_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "assert_embedding_model_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "assert_embedding_dim_consistent", lambda *args: None)
    monkeypatch.setattr(benchmark, "embed_texts", lambda *args, **kwargs: [[0.1, 0.2]])

    rows = []
    for rank in range(1, 13):
        if rank == 1:
            url, text = "https://example.com/overview.md", "Bearer access token overview"
        elif rank == 12:
            url, text = "https://example.com/auth.md", "Bearer access token setup details"
        else:
            url, text = f"https://example.com/other-{rank}.md", f"Unrelated documentation chunk {rank}"
        rows.append(
            (
                f"chunk-{rank}",
                text,
                {
                    "source": "api-docs",
                    "document": url.rsplit("/", 1)[-1],
                    "url": url,
                    "chunk_index": rank - 1,
                },
                rank / 100,
            )
        )

    def fake_search(_collection, _vector, top_k):
        selected = rows[:top_k]
        return {
            "ids": [[row[0] for row in selected]],
            "documents": [[row[1] for row in selected]],
            "metadatas": [[row[2] for row in selected]],
            "distances": [[row[3] for row in selected]],
        }

    monkeypatch.setattr(benchmark, "search_chunks", fake_search)

    report = benchmark.benchmark_retrieval(_config(), dataset_path, top_k=20)

    assert report["metrics"]["document_hit_rate_at_10"] == 0.0
    assert report["metrics"]["document_hit_rate_at_20"] == 1.0
    assert report["metrics"]["evidence_hit_rate_at_10"] == 0.0
    assert report["metrics"]["evidence_hit_rate_at_20"] == 1.0
    assert report["metrics"]["evidence_mrr_at_20"] == round(1 / 12, 4)
    assert report["metrics"]["expected_term_coverage_at_10"] == 0.0
    assert report["metrics"]["expected_term_coverage_at_20"] == 1.0
    assert report["cases"][0]["evidence_rank"] == 12
    assert report["cases"][0]["retrieved"][0]["is_evidence_chunk"] is False


def test_load_benchmark_dataset_rejects_invalid_relevant_chunk_terms(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "bad-group",
                        "query": "How does auth work?",
                        "relevant_urls": ["https://example.com/auth.md"],
                        "relevant_chunk_terms": [["bearer", 42]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_chunk_terms"):
        benchmark.load_benchmark_dataset(dataset_path)

def test_write_report_creates_parent_directories(tmp_path: Path) -> None:
    path = benchmark.write_report({"ok": True}, tmp_path / "reports" / "nested" / "report.json")

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
