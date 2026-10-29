from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caracal_rag.config import AppConfig, resolve_home
from caracal_rag.embeddings import embed_texts
from caracal_rag.manifest import assert_manifest_consistent, load_manifest
from caracal_rag.vectorstore import (
    _chroma_client,
    assert_embedding_dim_consistent,
    assert_embedding_model_consistent,
    search_chunks,
)

AUDIT_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 2
MAX_TOP_K = 20
DEFAULT_TOP_K = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(value: Any) -> str:
    return str(value or "").strip().casefold()


def _normalise_url(value: Any) -> str:
    return _normalise(value).rstrip("/")


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _collection_for(config: AppConfig):
    client = _chroma_client(
        host=config.chroma.host,
        port=config.chroma.port,
        ssl=config.chroma.ssl,
    )
    return client.get_collection(name=config.chroma.collection)


def audit_index(config: AppConfig) -> dict[str, Any]:
    """Inspect indexed chunks and metadata without calling the embedding API."""
    collection = _collection_for(config)
    response = collection.get(include=["documents", "metadatas"])
    ids = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []

    source_stats: dict[str, dict[str, Any]] = {}
    document_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    text_hash_counts: Counter[str] = Counter()
    embedding_models: Counter[str] = Counter()
    embedding_dimensions: Counter[str] = Counter()
    chunk_lengths: list[int] = []
    missing_url = 0
    missing_source = 0
    missing_document = 0
    missing_model = 0
    missing_dimension = 0
    empty_chunks = 0
    staging_chunks = 0
    committed_chunks = 0
    committed_urls: set[str] = set()

    for index, chunk_id in enumerate(ids):
        text = documents[index] if index < len(documents) else None
        metadata = metadatas[index] if index < len(metadatas) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        is_staging = bool(metadata.get("caracal_staging", False))
        if is_staging:
            staging_chunks += 1
            continue

        committed_chunks += 1
        source = str(metadata.get("source") or "")
        document = str(metadata.get("document") or "")
        url = str(metadata.get("url") or "")
        source_label = source or "<missing>"
        if url:
            committed_urls.add(url)
        if not url:
            missing_url += 1
        if not source:
            missing_source += 1
        if not document:
            missing_document += 1

        model = str(metadata.get("embedding_model") or "")
        if not model:
            missing_model += 1
        embedding_models[model or "<missing>"] += 1

        dimension = metadata.get("embedding_dim")
        if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
            embedding_dimensions[str(dimension)] += 1
        else:
            missing_dimension += 1
            embedding_dimensions["<missing-or-invalid>"] += 1

        chunk_text = text if isinstance(text, str) else ""
        length = len(chunk_text)
        chunk_lengths.append(length)
        if not chunk_text.strip():
            empty_chunks += 1
        else:
            text_hash_counts[hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()] += 1

        source_record = source_stats.setdefault(
            source_label,
            {"source": source_label, "chunks": 0, "documents": set(), "characters": 0},
        )
        source_record["chunks"] += 1
        source_record["characters"] += length
        document_key = (url or f"<missing-url:{chunk_id}>", source_label, document or "<missing>")
        doc_record = document_stats.setdefault(
            document_key,
            {
                "source": source_label,
                "document": document or "<missing>",
                "url": url,
                "chunks": 0,
                "characters": 0,
                "empty_chunks": 0,
                "chunk_indices": [],
            },
        )
        doc_record["chunks"] += 1
        doc_record["characters"] += length
        if not chunk_text.strip():
            doc_record["empty_chunks"] += 1
        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
            doc_record["chunk_indices"].append(chunk_index)
        if url:
            source_record["documents"].add(url)

    duplicate_groups = sum(1 for count in text_hash_counts.values() if count > 1)
    duplicate_rows = sum(count - 1 for count in text_hash_counts.values() if count > 1)
    warnings: list[str] = []
    if committed_chunks == 0:
        warnings.append("The collection has no committed chunks.")
    if staging_chunks:
        warnings.append(
            f"{staging_chunks} staging chunk(s) are present; indexing may have been interrupted."
        )
    if empty_chunks:
        warnings.append(f"{empty_chunks} committed chunk(s) have empty or whitespace-only text.")
    if missing_url or missing_source or missing_document:
        warnings.append("Some committed chunks are missing source/document/url metadata.")
    if missing_model:
        warnings.append(
            f"{missing_model} committed chunk(s) have no embedding_model metadata; "
            "the embedding space cannot be trusted."
        )
    if missing_dimension:
        warnings.append(
            f"{missing_dimension} committed chunk(s) have no valid embedding_dim metadata."
        )
    stored_models = set(embedding_models) - {"<missing>"}
    if stored_models and stored_models != {config.embedding.model}:
        warnings.append(
            "Stored embedding model metadata does not match the active configuration "
            f"({config.embedding.model!r}): {sorted(stored_models)!r}."
        )
    stored_dimensions = set(embedding_dimensions) - {"<missing-or-invalid>"}
    if len(stored_dimensions) > 1:
        warnings.append(
            "Mixed stored embedding dimensions detected: "
            f"{sorted(stored_dimensions)!r}."
        )
    if duplicate_groups:
        warnings.append(
            f"{duplicate_groups} exact-text duplicate group(s) found "
            f"({duplicate_rows} additional chunk row(s)). Review them; repeated text can be valid."
        )

    manifest_report: dict[str, Any]
    try:
        manifest = load_manifest(resolve_home().manifest_path)
        if manifest is None:
            manifest_report = {"status": "missing"}
            if committed_chunks:
                warnings.append("The local index manifest is missing.")
        else:
            matches_config = manifest.matches(config)
            manifest_report = {
                "status": "matches_config" if matches_config else "config_mismatch",
                "collection": manifest.collection,
                "embedding_model": manifest.embedding_model,
                "embedding_dim": manifest.embedding_dim,
                "document_count": manifest.document_count,
                "updated_at": manifest.updated_at,
                "matches_active_config": matches_config,
            }
            if not matches_config:
                warnings.append("The local index manifest does not match the active configuration.")
            numeric_dimensions = {
                int(dimension)
                for dimension in stored_dimensions
                if dimension.isdigit()
            }
            if manifest.embedding_dim is None and numeric_dimensions:
                warnings.append(
                    "The local manifest does not record embedding_dim; the "
                    f"committed collection vectors report {sorted(numeric_dimensions)}."
                )
            elif manifest.embedding_dim is not None and numeric_dimensions and (
                numeric_dimensions != {manifest.embedding_dim}
            ):
                warnings.append(
                    "The manifest embedding dimension does not match the committed "
                    f"collection vectors (manifest={manifest.embedding_dim}, "
                    f"collection={sorted(numeric_dimensions)})."
                )
            if committed_chunks and missing_url == 0 and (
                manifest.document_count != len(committed_urls)
            ):
                warnings.append(
                    "The manifest document count does not match the collection "
                    f"(manifest={manifest.document_count}, collection={len(committed_urls)})."
                )
    except Exception:
        manifest_report = {"status": "invalid_or_unreadable"}
        warnings.append("The local index manifest is invalid or unreadable.")

    source_output = []
    for source, record in sorted(source_stats.items()):
        source_output.append(
            {
                "source": source,
                "chunks": record["chunks"],
                "documents": len(record["documents"]),
                "characters": record["characters"],
            }
        )
    document_output = []
    for key, record in sorted(document_stats.items()):
        document_output.append(
            {
                **record,
                "chunk_indices": sorted(record["chunk_indices"]),
            }
        )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _now(),
        "collection": config.chroma.collection,
        "expected_embedding_model": config.embedding.model,
        "manifest": manifest_report,
        "counts": {
            "total_chunks": len(ids),
            "committed_chunks": committed_chunks,
            "staging_chunks": staging_chunks,
            "documents": len(committed_urls),
            "sources": len(source_stats),
            "empty_chunks": empty_chunks,
            "missing_url_metadata": missing_url,
            "missing_source_metadata": missing_source,
            "missing_document_metadata": missing_document,
            "missing_embedding_model_metadata": missing_model,
            "missing_embedding_dimension_metadata": missing_dimension,
            "duplicate_chunk_text_groups": duplicate_groups,
            "duplicate_chunk_text_rows": duplicate_rows,
        },
        "chunk_length_chars": {
            "min": min(chunk_lengths) if chunk_lengths else None,
            "median": statistics.median(chunk_lengths) if chunk_lengths else None,
            "mean": round(statistics.mean(chunk_lengths), 2) if chunk_lengths else None,
            "p95": _percentile(chunk_lengths, 0.95),
            "max": max(chunk_lengths) if chunk_lengths else None,
            "over_1800_chars": sum(length > 1800 for length in chunk_lengths),
        },
        "embedding_models": dict(sorted(embedding_models.items())),
        "embedding_dimensions": dict(sorted(embedding_dimensions.items())),
        "by_source": source_output,
        "documents": document_output,
        "warnings": warnings,
    }


def load_benchmark_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an explicitly curated, versioned retrieval test set."""
    dataset_path = Path(path)
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark dataset {dataset_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Benchmark dataset must be a JSON object with version: 1.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark dataset must contain a non-empty 'cases' array.")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case at index {index} must be a JSON object.")
        case_id = case.get("id")
        query = case.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Benchmark case at index {index} needs a non-empty string id.")
        case_id = case_id.strip()
        if case_id in seen_ids:
            raise ValueError(f"Benchmark case id {case_id!r} is duplicated.")
        seen_ids.add(case_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark case {case_id!r} needs a non-empty string query.")
        targets: dict[str, list[str]] = {}
        for field in ("relevant_urls", "relevant_documents", "relevant_sources"):
            values = case.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError(f"{field} in benchmark case {case_id!r} must be a string array.")
            targets[field] = list(dict.fromkeys(item.strip() for item in values if item.strip()))
        if not any(targets.values()):
            raise ValueError(
                f"Benchmark case {case_id!r} needs at least one of relevant_urls, "
                "relevant_documents, or relevant_sources."
            )
        terms = case.get("expected_terms", [])
        if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
            raise ValueError(
                f"expected_terms in benchmark case {case_id!r} must be a string array."
            )
        relevant_chunk_terms = case.get("relevant_chunk_terms", [])
        if not isinstance(relevant_chunk_terms, list) or any(
            not isinstance(group, list)
            or not group
            or any(not isinstance(term, str) or not term.strip() for term in group)
            for group in relevant_chunk_terms
        ):
            raise ValueError(
                f"relevant_chunk_terms in benchmark case {case_id!r} must be an array "
                "of non-empty string arrays. Every group must be supported by one chunk."
            )
        validated.append(
            {
                "id": case_id,
                "query": query.strip(),
                **targets,
                "expected_terms": list(
                    dict.fromkeys(term.strip() for term in terms if term.strip())
                ),
                "relevant_chunk_terms": [
                    list(dict.fromkeys(term.strip() for term in group))
                    for group in relevant_chunk_terms
                ],
            }
        )
    return validated


def _relevance_basis(case: dict[str, Any]) -> tuple[str, list[str]]:
    for field in ("relevant_urls", "relevant_documents", "relevant_sources"):
        values = case[field]
        if values:
            return field, values
    raise ValueError(f"Benchmark case {case['id']!r} has no relevance labels.")


def _result_target(result: dict[str, Any], field: str) -> str:
    if field == "relevant_urls":
        return _normalise_url(result.get("url"))
    if field == "relevant_documents":
        return _normalise(result.get("document"))
    return _normalise(result.get("source"))


def _query_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    def first_group(field: str) -> list[Any]:
        value = response.get(field) or []
        if value and isinstance(value[0], list):
            return value[0]
        return value

    ids = first_group("ids")
    documents = first_group("documents")
    metadatas = first_group("metadatas")
    distances = first_group("distances")
    results: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        content = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        results.append(
            {
                "rank": index + 1,
                "chunk_id": str(chunk_id),
                "source": str(metadata.get("source") or ""),
                "document": str(metadata.get("document") or ""),
                "url": str(metadata.get("url") or ""),
                "chunk_index": metadata.get("chunk_index"),
                "distance": float(distance) if isinstance(distance, (int, float)) else None,
                "content": str(content or ""),
                "content_preview": str(content or "")[:240],
            }
        )
    return results


def benchmark_retrieval(
    config: AppConfig,
    dataset_path: str | Path,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Evaluate document retrieval and chunk-level evidence; no chat LLM is used."""
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}.")
    cases = load_benchmark_dataset(dataset_path)
    collection = _collection_for(config)
    assert_manifest_consistent(resolve_home().manifest_path, config)
    assert_embedding_model_consistent(collection, config.embedding.model)

    case_results: list[dict[str, Any]] = []
    total_query_ms = 0.0
    for case in cases:
        started = time.perf_counter()
        vectors = embed_texts(
            [case["query"]],
            api_base=config.embedding.api_base,
            api_key=config.embedding.api_key,
            model=config.embedding.model,
        )
        if not vectors or not vectors[0]:
            raise RuntimeError(
                f"Embedding endpoint returned an empty vector for case {case['id']!r}."
            )
        query_embedding = vectors[0]
        assert_embedding_dim_consistent(collection, len(query_embedding))
        response = search_chunks(collection, query_embedding, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_query_ms += elapsed_ms

        retrieved = _query_results(response)
        relevance_field, expected = _relevance_basis(case)
        expected_targets = {
            _normalise_url(value)
            if relevance_field == "relevant_urls"
            else _normalise(value)
            for value in expected
        }
        expected_terms = case["expected_terms"]
        # Explicit groups declare terms that must co-occur in one returned chunk.
        # For datasets without groups, each expected term is an independent evidence
        # signal, preserving a useful diagnostic for older/local benchmark files.
        relevant_chunk_terms = case["relevant_chunk_terms"] or [
            [term] for term in expected_terms
        ]
        found_targets: set[str] = set()
        found_terms_in_target_chunks: set[str] = set()
        found_chunk_groups: set[int] = set()
        target_ranks: list[int] = []
        evidence_ranks: list[int] = []

        for result in retrieved:
            target = _result_target(result, relevance_field)
            is_target_chunk = bool(target and target in expected_targets)
            chunk_text = result["content"].casefold()
            terms_in_chunk = [
                term for term in expected_terms if term.casefold() in chunk_text
            ]
            matched_groups = [
                group_index
                for group_index, group in enumerate(relevant_chunk_terms)
                if all(term.casefold() in chunk_text for term in group)
            ] if is_target_chunk else []

            result["matches_expected_target"] = is_target_chunk
            result["found_expected_terms"] = terms_in_chunk
            result["matched_relevant_chunk_groups"] = matched_groups
            result["is_evidence_chunk"] = bool(matched_groups)

            if is_target_chunk:
                found_targets.add(target)
                target_ranks.append(result["rank"])
                found_terms_in_target_chunks.update(terms_in_chunk)
                found_chunk_groups.update(matched_groups)
            if matched_groups:
                evidence_ranks.append(result["rank"])

        target_chunk_coverage = (
            len(found_terms_in_target_chunks) / len(expected_terms)
            if expected_terms
            else None
        )
        chunk_group_coverage = (
            len(found_chunk_groups) / len(relevant_chunk_terms)
            if relevant_chunk_terms
            else None
        )
        for result in retrieved:
            result.pop("content", None)

        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "relevance_basis": relevance_field,
                "expected_targets": expected,
                "found_targets": sorted(found_targets),
                "document_hit": bool(target_ranks),
                "document_reciprocal_rank": 1.0 / min(target_ranks) if target_ranks else 0.0,
                "document_recall": (
                    len(found_targets) / len(expected_targets) if expected_targets else 0.0
                ),
                "evidence_hit": bool(evidence_ranks),
                "evidence_rank": min(evidence_ranks) if evidence_ranks else None,
                "expected_terms": expected_terms,
                "found_expected_terms_in_target_chunks": [
                    term for term in expected_terms if term in found_terms_in_target_chunks
                ],
                "expected_term_coverage": target_chunk_coverage,
                "relevant_chunk_terms": relevant_chunk_terms,
                "matched_relevant_chunk_groups": sorted(found_chunk_groups),
                "relevant_chunk_group_coverage": chunk_group_coverage,
                "latency_ms": round(elapsed_ms, 2),
                "retrieved": retrieved,
            }
        )

    cutoffs = sorted({cutoff for cutoff in (1, 3, 5, 10, top_k) if cutoff <= top_k})
    metrics: dict[str, Any] = {}
    for cutoff in cutoffs:
        document_hits: list[bool] = []
        document_recalls: list[float] = []
        document_reciprocal_ranks: list[float] = []
        evidence_hits: list[bool] = []
        evidence_reciprocal_ranks: list[float] = []
        chunk_group_coverages: list[float] = []
        matched_expected_terms = 0
        total_expected_terms = 0

        for case in case_results:
            retrieved = case["retrieved"][:cutoff]
            relevance_field = case["relevance_basis"]
            expected_targets = {
                _normalise_url(value)
                if relevance_field == "relevant_urls"
                else _normalise(value)
                for value in case["expected_targets"]
            }
            found: set[str] = set()
            first_target_rank: int | None = None
            first_evidence_rank: int | None = None
            found_groups: set[int] = set()
            found_terms: set[str] = set()

            for result in retrieved:
                target = _result_target(result, relevance_field)
                if target and target in expected_targets:
                    found.add(target)
                    if first_target_rank is None:
                        first_target_rank = result["rank"]
                    found_terms.update(result["found_expected_terms"])
                    found_groups.update(result["matched_relevant_chunk_groups"])
                if result["is_evidence_chunk"] and first_evidence_rank is None:
                    first_evidence_rank = result["rank"]

            document_hits.append(bool(found))
            document_recalls.append(
                len(found) / len(expected_targets) if expected_targets else 0.0
            )
            document_reciprocal_ranks.append(
                1.0 / first_target_rank if first_target_rank else 0.0
            )
            evidence_hits.append(first_evidence_rank is not None)
            evidence_reciprocal_ranks.append(
                1.0 / first_evidence_rank if first_evidence_rank else 0.0
            )

            all_terms = case["expected_terms"]
            matched_expected_terms += sum(term in found_terms for term in all_terms)
            total_expected_terms += len(all_terms)
            group_count = len(case["relevant_chunk_terms"])
            if group_count:
                chunk_group_coverages.append(len(found_groups) / group_count)

        metrics[f"document_hit_rate_at_{cutoff}"] = round(
            sum(document_hits) / len(document_hits), 4
        )
        metrics[f"document_recall_at_{cutoff}"] = round(
            sum(document_recalls) / len(document_recalls), 4
        )
        metrics[f"document_mrr_at_{cutoff}"] = round(
            sum(document_reciprocal_ranks) / len(document_reciprocal_ranks), 4
        )
        metrics[f"evidence_hit_rate_at_{cutoff}"] = round(
            sum(evidence_hits) / len(evidence_hits), 4
        )
        metrics[f"evidence_mrr_at_{cutoff}"] = round(
            sum(evidence_reciprocal_ranks) / len(evidence_reciprocal_ranks), 4
        )
        metrics[f"relevant_chunk_group_coverage_at_{cutoff}"] = (
            round(sum(chunk_group_coverages) / len(chunk_group_coverages), 4)
            if chunk_group_coverages
            else None
        )
        metrics[f"expected_term_coverage_at_{cutoff}"] = (
            round(matched_expected_terms / total_expected_terms, 4)
            if total_expected_terms
            else None
        )

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "generated_at": _now(),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "collection": config.chroma.collection,
        "embedding_model": config.embedding.model,
        "top_k": top_k,
        "case_count": len(case_results),
        "mean_latency_ms": round(total_query_ms / len(case_results), 2),
        "expected_term_coverage": metrics[f"expected_term_coverage_at_{top_k}"],
        "metrics": metrics,
        "cases": case_results,
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a local JSON report, creating parent directories when needed."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
