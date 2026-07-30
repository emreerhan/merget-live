from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .answering import AnswerGenerator
from .benchmark import (
    QueryBenchmarkBuilder,
    benchmark_summary,
    build_candidate_pool,
    load_judgments,
    load_queries,
    pool_candidates,
)
from .calibration import (
    annotate_and_filter_hits,
    answerability_probability,
    fit_calibration,
    query_features,
)
from .config import AppConfig, load_config
from .embeddings import DenseEncoder, select_index_records
from .evidence import EvidenceBuilder, evidence_quality, load_evidence, write_evidence
from .evaluation import answer_metrics, retrieval_metrics
from .io import DatasetReader
from .lexical import BM25SparseIndex
from .models import AnswerRecord, RetrievalHit, TemporalMode
from .openrouter import OpenRouterClient
from .partition import assign_partitions, select_pilot
from .provenance import (
    assert_unchanged,
    build_manifest,
    canonical_hash,
    code_tree_hash,
    sha256_file,
    sha256_path,
    snapshot_metadata,
    source_files,
    write_json_atomic,
)
from .retrieval import HybridRetriever, classify_temporal_mode
from .security import secure_key_file
from .training import assert_training_isolation, build_training_examples, train_encoder


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _client(config: AppConfig, stage: str) -> OpenRouterClient:
    return OpenRouterClient(
        config.openrouter,
        config.key_file,
        config.derived_dir / "openrouter-cache" / stage,
    )


def _paths(config: AppConfig) -> dict[str, Path]:
    return {
        "evidence": config.derived_dir / "evidence" / "records.jsonl",
        "quality": config.derived_dir / "evidence" / "quality.json",
        "partition": config.derived_dir / "evidence" / "partitions.json",
        "queries": config.derived_dir / "benchmark" / "accepted-queries.jsonl",
        "judgments": config.derived_dir / "benchmark" / "relevance-judgments.jsonl",
        "vectors": config.derived_dir / "embeddings" / "vectors.npz",
        "vectors_fine": config.derived_dir / "embeddings" / "vectors-fine-tuned.npz",
        "lexical": config.derived_dir / "index" / "lexical.json",
        "qdrant": config.derived_dir / "index" / "qdrant",
        "qdrant_fine": config.derived_dir / "index" / "qdrant-fine-tuned",
        "model": config.derived_dir / "models" / "bge-small-entireio",
        "calibration": config.derived_dir / "evaluation" / "score-calibration.json",
    }


def _record_manifest(
    config: AppConfig,
    stage: str,
    *,
    input_paths,
    artifact_paths,
    model_settings: dict | None = None,
    counts: dict | None = None,
) -> str:
    inputs = [Path(path) for path in input_paths if Path(path).is_file()]
    artifacts = [Path(path) for path in artifact_paths if Path(path).exists()]
    manifest = build_manifest(
        stage=stage,
        input_paths=inputs,
        config=config.model_dump(mode="json"),
        model_settings=model_settings,
        code_version=code_tree_hash(config.data_dir),
    )
    manifest.artifact_hashes.update(
        {
            str(path.relative_to(config.data_dir)): sha256_path(path)
            for path in artifacts
        }
    )
    manifest.counts.update(counts or {})
    write_json_atomic(
        config.derived_dir / "manifests" / f"{manifest.run_id}.json",
        manifest,
    )
    return manifest.run_id


def _cached_api_usage(cache_root: Path) -> dict:
    total = Counter()
    by_stage = {}
    if not cache_root.is_dir():
        return {"total": {}, "by_stage": {}}
    for stage_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        stage = Counter()
        for path in stage_dir.rglob("*.json"):
            try:
                usage = json.loads(path.read_text(encoding="utf-8")).get("usage", {})
            except (json.JSONDecodeError, OSError):
                continue
            stage["requests"] += 1
            stage["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            stage["output_tokens"] += int(usage.get("completion_tokens") or 0)
            stage["cost"] += float(usage.get("cost") or 0.0)
        if stage:
            by_stage[stage_dir.name] = dict(stage)
            total.update(stage)
    return {"total": dict(total), "by_stage": by_stage}


def _required_index_ids(paths: dict[str, Path]) -> set[str]:
    if not paths["queries"].is_file():
        return set()
    queries = load_queries(paths["queries"])
    return {
        evidence_id
        for query in queries
        for evidence_id in (
            query.primary_positive_ids
            + query.supporting_positive_ids
            + query.generation_source_ids
        )
    }


def command_validate(config: AppConfig, args) -> dict:
    if args.fix_key_permissions:
        secure_key_file(config.key_file, fix=True)
    secure_key_file(config.key_file)
    counts = DatasetReader(config.data_dir).validate()
    result = {"valid": True, "counts": counts, "data_dir": str(config.data_dir)}
    print(json.dumps(result, indent=2))
    return result


def command_prepare(config: AppConfig, args) -> dict:
    reader = DatasetReader(config.data_dir)
    reader.validate()
    immutable = snapshot_metadata(source_files(config.data_dir))
    session_rows = [
        row for _, row in reader.iter_table(
            "sessions",
            columns=[
                "session_id", "created_at", "agent", "user_id",
                "files_touched_count", "checkpoints_count",
            ],
        )
    ]
    selected = None if args.full else select_pilot(session_rows, config.pilot)
    records = EvidenceBuilder(reader, config.evidence).build(selected)
    records, partition_report = assign_partitions(records, config.pilot)
    paths = _paths(config)
    write_evidence(paths["evidence"], records)
    quality = evidence_quality(records)
    write_json_atomic(paths["quality"], quality)
    write_json_atomic(paths["partition"], partition_report)
    assert_unchanged(immutable)
    manifest_id = _record_manifest(
        config,
        "prepare",
        input_paths=source_files(config.data_dir),
        artifact_paths=[paths["evidence"], paths["quality"], paths["partition"]],
        counts={
            "selected_sessions": len(selected or session_rows),
            "evidence": len(records),
        },
    )
    result = {"quality": quality, "partitions": partition_report, "manifest": manifest_id}
    print(json.dumps(result, indent=2))
    return result


def command_generate(config: AppConfig, args) -> dict:
    records = load_evidence(_paths(config)["evidence"])
    client = _client(config, "benchmark")
    builder = QueryBenchmarkBuilder(
        client, records, config.derived_dir / "benchmark", seed=config.pilot.seed
    )
    generated = builder.generate(per_group=args.per_group, max_groups=args.max_groups)
    accepted, assessments = builder.criticize(
        generated, double_evaluation=args.double_evaluation
    )
    summary = benchmark_summary(accepted, assessments)
    summary["usage"] = client.usage
    write_json_atomic(config.derived_dir / "benchmark" / "summary.json", summary)
    _record_manifest(
        config,
        "generate",
        input_paths=[_paths(config)["evidence"]],
        artifact_paths=[
            config.derived_dir / "benchmark" / name
            for name in (
                "generated-queries.jsonl",
                "accepted-queries.jsonl",
                "critic-assessments.json",
                "generation-failures.json",
                "summary.json",
            )
        ],
        model_settings=config.openrouter.model_dump(mode="json"),
        counts={
            "generated": len(generated),
            "accepted": len(accepted),
            "assessments": len(assessments),
        },
    )
    print(json.dumps(summary, indent=2))
    return summary


def command_judge(config: AppConfig, args) -> dict:
    paths = _paths(config)
    records = select_index_records(
        load_evidence(paths["evidence"]),
        config.embedding,
        _required_index_ids(paths),
    )
    queries = load_queries(paths["queries"])
    encoder = DenseEncoder(config.embedding)
    vectors = encoder.encode_records(records, paths["vectors"])
    lexical = (
        BM25SparseIndex.load(paths["lexical"])
        if paths["lexical"].is_file()
        else BM25SparseIndex()
    )
    if set(lexical.doc_vectors) != {record.evidence_id for record in records}:
        lexical.fit(records)
    lexical.save(paths["lexical"])
    pool_path = config.derived_dir / "benchmark" / "candidate-pools.json"
    if pool_path.is_file() and not args.rebuild_pools:
        traces = json.loads(pool_path.read_text(encoding="utf-8"))
        candidates = {
            query.query_id: pool_candidates(
                *(
                    {
                        query.query_id: traces.get(query.query_id, {}).get(
                            source, []
                        )
                    }
                    for source in (
                        "bm25",
                        "pretrained_dense",
                        "metadata_neighbor",
                        "temporally_diverse",
                    )
                ),
                limit=args.limit,
            ).get(query.query_id, [])
            for query in queries
        }
    else:
        candidates, traces = build_candidate_pool(
            queries, records, encoder, vectors, lexical, limit=args.limit
        )
        write_json_atomic(pool_path, traces)
    client = _client(config, "judgments")
    judgments = QueryBenchmarkBuilder(
        client, records, config.derived_dir / "benchmark", seed=config.pilot.seed
    ).judge_candidates(queries, candidates)
    summary = benchmark_summary(queries, [], judgments)
    summary["usage"] = client.usage
    write_json_atomic(config.derived_dir / "benchmark" / "judgment-summary.json", summary)
    _record_manifest(
        config,
        "judge",
        input_paths=[paths["evidence"], paths["queries"], paths["vectors"]],
        artifact_paths=[
            paths["judgments"],
            config.derived_dir / "benchmark" / "candidate-pools.json",
            config.derived_dir / "benchmark" / "relevance-failures.json",
            config.derived_dir / "benchmark" / "judgment-summary.json",
            paths["lexical"],
        ],
        model_settings=config.openrouter.model_dump(mode="json"),
        counts={"queries": len(queries), "judgments": len(judgments)},
    )
    print(json.dumps(summary, indent=2))
    return summary


def _load_index(config: AppConfig, model_path: Path | None = None):
    paths = _paths(config)
    records = select_index_records(
        load_evidence(paths["evidence"]),
        config.embedding,
        _required_index_ids(paths),
    )
    encoder = DenseEncoder(config.embedding, model_path=model_path)
    vectors = encoder.encode_records(
        records,
        paths["vectors_fine"] if model_path else paths["vectors"],
    )
    if paths["lexical"].is_file():
        lexical = BM25SparseIndex.load(paths["lexical"])
    else:
        lexical = BM25SparseIndex()
    if set(lexical.doc_vectors) != {record.evidence_id for record in records}:
        lexical.fit(records)
        lexical.save(paths["lexical"])
    retriever = HybridRetriever(
        records,
        vectors,
        lexical,
        config.retrieval,
        paths["qdrant_fine"] if model_path else paths["qdrant"],
    )
    return records, encoder, retriever


def command_index(config: AppConfig, args) -> dict:
    paths = _paths(config)
    model_path = paths["model"] if args.fine_tuned else None
    records, encoder, retriever = _load_index(config, model_path=model_path)
    retriever.build(recreate=args.recreate)
    result = {"records": len(records), "dimension": encoder.dimension, "collection": config.retrieval.collection}
    _record_manifest(
        config,
        "index-fine-tuned" if args.fine_tuned else "index-pretrained",
        input_paths=[
            paths["evidence"],
            paths["lexical"],
            paths["vectors_fine"] if args.fine_tuned else paths["vectors"],
            *(
                [paths["model"] / "model.safetensors"]
                if args.fine_tuned
                else []
            ),
        ],
        artifact_paths=[
            paths["qdrant_fine"] if args.fine_tuned else paths["qdrant"]
        ],
        model_settings={
            **config.embedding.model_dump(mode="json"),
            "fine_tuned": args.fine_tuned,
        },
        counts={"records": len(records), "dimension": encoder.dimension},
    )
    print(json.dumps(result, indent=2))
    return result


def command_query(config: AppConfig, args) -> list[dict]:
    model_path = _paths(config)["model"] if args.fine_tuned else None
    records, encoder, retriever = _load_index(config, model_path=model_path)
    vector = encoder.encode_queries([args.query])[0]
    mode = TemporalMode(args.temporal_mode) if args.temporal_mode else classify_temporal_mode(args.query)
    hits = retriever.search(args.query, vector, temporal_mode=mode)
    if not args.fine_tuned and not args.no_calibration:
        hits, _ = _apply_runtime_calibration(
            config, args.query, vector, retriever, hits
        )
    result = [hit.model_dump(mode="json") for hit in hits]
    print(json.dumps(result, indent=2))
    return result


def command_answer(config: AppConfig, args) -> dict:
    records, encoder, retriever = _load_index(
        config, model_path=_paths(config)["model"] if args.fine_tuned else None
    )
    vector = encoder.encode_queries([args.query])[0]
    mode = TemporalMode(args.temporal_mode) if args.temporal_mode else classify_temporal_mode(args.query)
    hits = retriever.search(args.query, vector, temporal_mode=mode)
    answerable_probability = None
    if not args.fine_tuned and not args.no_calibration:
        hits, answerable_probability = _apply_runtime_calibration(
            config, args.query, vector, retriever, hits
        )
        calibration = _load_calibration(config)
        if (
            calibration is not None
            and calibration["answerability"].get(
                "enabled_for_abstention", False
            )
            and answerable_probability
            < calibration["answerability"]["probability_threshold"]
        ):
            hits = []
    result = AnswerGenerator(
        _client(config, "answers"),
        records,
        config.derived_dir / "answers",
    ).answer(args.query, hits, temporal_mode=mode)
    print(result.model_dump_json(indent=2))
    return result.model_dump(mode="json")


def _load_calibration(config: AppConfig) -> dict | None:
    path = _paths(config)["calibration"]
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _apply_runtime_calibration(
    config: AppConfig,
    query_text: str,
    query_vector: np.ndarray,
    retriever: HybridRetriever,
    hits: list[RetrievalHit],
) -> tuple[list[RetrievalHit], float | None]:
    artifact = _load_calibration(config)
    if artifact is None:
        return hits, None
    ids = sorted(retriever.dense_vectors)
    matrix = np.stack([retriever.dense_vectors[evidence_id] for evidence_id in ids])
    scores = matrix @ query_vector
    order = np.argsort(-scores, kind="stable")[:50]
    dense_ids = [ids[index] for index in order]
    dense_scores = [float(scores[index]) for index in order]
    bm25_ids = [
        evidence_id
        for evidence_id, _ in retriever.lexical.rank(query_text, limit=50)
    ]
    features = query_features(
        dense_scores,
        dense_ids,
        bm25_ids,
        artifact,
    )
    probability = answerability_probability(features, artifact)
    filtered = annotate_and_filter_hits(hits, artifact)
    return [
        hit.model_copy(
            update={
                "payload": {
                    **hit.payload,
                    "answerability_probability": probability,
                    "answerability_threshold": artifact["answerability"][
                        "probability_threshold"
                    ],
                }
            }
        )
        for hit in filtered
    ], probability


def _baseline_hits(
    query,
    query_vector,
    records,
    retriever,
    *,
    variant: str,
    limit: int = 50,
):
    allowed = [
        record
        for record in records
        if record.partition == query.partition
        and (
            query.as_of is None
            or (
                record.timestamp is not None
                and record.timestamp <= query.as_of
            )
        )
    ]
    by_id = {record.evidence_id: record for record in allowed}
    if variant.endswith("_dense"):
        ids = sorted(set(by_id) & retriever.dense_vectors.keys())
        matrix = np.stack([retriever.dense_vectors[evidence_id] for evidence_id in ids])
        scores = matrix @ query_vector
        ranked = [
            (ids[index], float(scores[index]))
            for index in np.argsort(-scores, kind="stable")[:limit]
        ]
        return [
            RetrievalHit(
                evidence_id=evidence_id,
                dense_rank=rank,
                dense_score=score,
                fusion_score=score,
                final_score=score,
                citations=by_id[evidence_id].source_refs,
                payload={
                    "title": by_id[evidence_id].title,
                    "timestamp": (
                        by_id[evidence_id].timestamp.isoformat()
                        if by_id[evidence_id].timestamp
                        else None
                    ),
                    "temporal_mode": query.temporal_mode.value,
                    "filters": {"partition": query.partition},
                    "group_member_ids": [evidence_id],
                    "final_rank": rank,
                },
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    if variant == "bm25":
        ranked = [
            (evidence_id, score)
            for evidence_id, score in retriever.lexical.rank(query.query, limit=200)
            if evidence_id in by_id
        ][:limit]
        return [
            RetrievalHit(
                evidence_id=evidence_id,
                lexical_rank=rank,
                lexical_score=score,
                fusion_score=score,
                final_score=score,
                citations=by_id[evidence_id].source_refs,
                payload={
                    "title": by_id[evidence_id].title,
                    "timestamp": (
                        by_id[evidence_id].timestamp.isoformat()
                        if by_id[evidence_id].timestamp
                        else None
                    ),
                    "temporal_mode": query.temporal_mode.value,
                    "filters": {"partition": query.partition},
                    "group_member_ids": [evidence_id],
                    "final_rank": rank,
                },
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    raise ValueError(f"unsupported baseline variant: {variant}")


def command_answer_benchmark(config: AppConfig, args) -> dict:
    paths = _paths(config)
    frozen_path = config.derived_dir / "evaluation" / "frozen-config.json"
    frozen = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.is_file()
        else {}
    )
    selected_variant = frozen.get(
        "model_variant",
        "fine_tuned_hybrid" if args.fine_tuned else "pretrained_hybrid",
    )
    uses_fine_tuned = selected_variant.startswith("fine_tuned")
    records, encoder, retriever = _load_index(
        config, model_path=paths["model"] if uses_fine_tuned else None
    )
    if frozen:
        retriever.config = retriever.config.model_copy(
            update={
                "relevance_floor": frozen["relevance_floor"],
                "temporal_weight": frozen["temporal_weight"],
            }
        )
    queries = [
        query
        for query in load_queries(paths["queries"])
        if query.partition == args.partition
    ]
    query_vectors = encoder.encode_queries([query.query for query in queries])
    hits_by_id = {}
    for query, vector in zip(queries, query_vectors, strict=True):
        if selected_variant in {"bm25", "pretrained_dense", "fine_tuned_dense"}:
            hits_by_id[query.query_id] = _baseline_hits(
                query,
                vector,
                records,
                retriever,
                variant=selected_variant,
            )
        else:
            hits_by_id[query.query_id] = retriever.search(
                query.query,
                vector,
                temporal_mode=query.temporal_mode,
                as_of=query.as_of,
                filters={"partition": query.partition},
                limit=50,
            )
    client = _client(config, "answers")
    generator = AnswerGenerator(client, records, config.derived_dir / "answers")

    def generate(query):
        return generator.answer(
            query.query,
            hits_by_id[query.query_id],
            query_id=query.query_id,
            temporal_mode=query.temporal_mode,
        )

    answers, failures = [], []
    with ThreadPoolExecutor(max_workers=config.openrouter.max_concurrency) as executor:
        futures = {executor.submit(generate, query): query for query in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                answers.append(future.result())
            except Exception as exc:
                failures.append({"query_id": query.query_id, "error": str(exc)})
    result = {
        "partition": args.partition,
        "model": selected_variant,
        "requested": len(queries),
        "completed": len(answers),
        "valid": sum(answer.valid for answer in answers),
        "failures": failures,
        "usage": client.usage,
    }
    write_json_atomic(
        config.derived_dir / "evaluation" / f"answer-run-{args.partition}.json",
        result,
    )
    _record_manifest(
        config,
        f"answer-{args.partition}",
        input_paths=[
            paths["evidence"],
            paths["queries"],
            frozen_path,
            paths["vectors_fine"] if uses_fine_tuned else paths["vectors"],
        ],
        artifact_paths=[
            config.derived_dir / "answers",
            config.derived_dir / "evaluation" / f"answer-run-{args.partition}.json",
        ],
        model_settings={
            **config.openrouter.model_dump(mode="json"),
            "retrieval_model": result["model"],
        },
        counts={
            "requested": result["requested"],
            "completed": result["completed"],
            "valid": result["valid"],
        },
    )
    print(json.dumps(result, indent=2))
    return result


def command_train(config: AppConfig, args) -> dict:
    paths = _paths(config)
    records = load_evidence(paths["evidence"])
    queries = load_queries(paths["queries"])
    judgments = load_judgments(paths["judgments"])
    examples = build_training_examples(queries, records, judgments)
    validation_examples = build_training_examples(
        queries,
        records,
        judgments,
        partitions={"validation"},
    )
    assert_training_isolation(examples, queries, records)
    report = train_encoder(
        examples,
        validation_examples,
        config.embedding,
        config.training,
        paths["model"],
    )
    _record_manifest(
        config,
        "train",
        input_paths=[paths["evidence"], paths["queries"], paths["judgments"]],
        artifact_paths=[paths["model"]],
        model_settings=config.embedding.model_dump(mode="json"),
        counts={
            "examples": report["examples"],
            "completed_epochs": report["completed_epochs"],
        },
    )
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(config: AppConfig, args) -> dict:
    paths = _paths(config)
    records, encoder, retriever = _load_index(
        config, model_path=paths["model"] if args.fine_tuned else None
    )
    queries = load_queries(paths["queries"])
    judgments = load_judgments(paths["judgments"])
    selected = [query for query in queries if query.partition == args.partition]
    rankings = {}
    for query in selected:
        vector = encoder.encode_queries([query.query])[0]
        rankings[query.query_id] = retriever.search(
            query.query,
            vector,
            temporal_mode=query.temporal_mode,
            as_of=query.as_of,
            limit=50,
        )
    result = {"retrieval": retrieval_metrics(rankings, judgments, selected)}
    answers_path = config.derived_dir / "answers"
    if answers_path.is_dir():
        answers = [
            AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in answers_path.glob("*.json")
        ]
        result["answers"] = answer_metrics(answers, selected, judgments)
    output = config.derived_dir / "evaluation" / f"{args.partition}.json"
    write_json_atomic(output, result)
    print(json.dumps(result["retrieval"]["aggregate"], indent=2))
    return result


def _experiment_rankings(queries, encoder, retriever, lexical, records):
    by_id = {record.evidence_id: record for record in records}
    allowed_by_partition = {
        partition: {
            record.evidence_id
            for record in records
            if record.partition == partition
        }
        for partition in ("train", "validation", "evaluation")
    }
    dense_by_partition = {}
    for partition, allowed in allowed_by_partition.items():
        ids = sorted(allowed & retriever.dense_vectors.keys())
        dense_by_partition[partition] = (
            ids,
            np.stack([retriever.dense_vectors[evidence_id] for evidence_id in ids]),
        )
    rankings = {"bm25": {}, "dense": {}, "hybrid": {}, "hybrid_temporal": {}}
    original_config = retriever.config
    query_vectors = encoder.encode_queries([query.query for query in queries])
    for query, query_vector in zip(queries, query_vectors, strict=True):
        allowed = allowed_by_partition[query.partition]
        lexical_rows = [
            (evidence_id, score)
            for evidence_id, score in lexical.rank(query.query, limit=200)
            if evidence_id in allowed
        ][:50]
        rankings["bm25"][query.query_id] = [
            RetrievalHit(
                evidence_id=evidence_id,
                lexical_rank=rank,
                lexical_score=score,
                fusion_score=score,
                final_score=score,
                citations=by_id[evidence_id].source_refs,
                payload={
                    "temporal_mode": query.temporal_mode.value,
                    "filters": {"partition": query.partition},
                    "timestamp": (
                        by_id[evidence_id].timestamp.isoformat()
                        if by_id[evidence_id].timestamp
                        else None
                    ),
                },
            )
            for rank, (evidence_id, score) in enumerate(lexical_rows, start=1)
        ]
        dense_ids, dense_matrix = dense_by_partition[query.partition]
        order = np.argsort(-(dense_matrix @ query_vector), kind="stable")[:50]
        rankings["dense"][query.query_id] = [
            RetrievalHit(
                evidence_id=dense_ids[index],
                dense_rank=rank,
                dense_score=float(dense_matrix[index] @ query_vector),
                fusion_score=float(dense_matrix[index] @ query_vector),
                final_score=float(dense_matrix[index] @ query_vector),
                citations=by_id[dense_ids[index]].source_refs,
                payload={
                    "temporal_mode": query.temporal_mode.value,
                    "filters": {"partition": query.partition},
                    "timestamp": (
                        by_id[dense_ids[index]].timestamp.isoformat()
                        if by_id[dense_ids[index]].timestamp
                        else None
                    ),
                },
            )
            for rank, index in enumerate(order, start=1)
        ]
        retriever.config = original_config.model_copy(update={"temporal_weight": 0.0})
        rankings["hybrid"][query.query_id] = retriever.search(
            query.query,
            query_vector,
            temporal_mode=query.temporal_mode,
            as_of=query.as_of,
            filters={"partition": query.partition},
            limit=50,
        )
        retriever.config = original_config
        rankings["hybrid_temporal"][query.query_id] = retriever.search(
            query.query,
            query_vector,
            temporal_mode=query.temporal_mode,
            as_of=query.as_of,
            filters={"partition": query.partition},
            limit=50,
        )
    retriever.config = original_config
    return rankings


def _trace_rankings(rankings: dict, variant: str) -> dict:
    traced = {}
    for query_id, values in rankings.items():
        rows = []
        for rank, value in enumerate(values, start=1):
            if isinstance(value, str):
                rows.append(
                    {
                        "evidence_id": value,
                        "final_rank": rank,
                        f"{variant}_rank": rank,
                    }
                )
                continue
            rows.append(
                {
                    "evidence_id": value.evidence_id,
                    "final_rank": rank,
                    "dense_rank": value.dense_rank,
                    "dense_score": value.dense_score,
                    "lexical_rank": value.lexical_rank,
                    "lexical_score": value.lexical_score,
                    "fusion_score": value.fusion_score,
                    "temporal_score": value.temporal_score,
                    "final_score": value.final_score,
                    "group_id": value.group_id,
                    "group_member_ids": value.payload.get("group_member_ids", []),
                    "temporal_mode": value.payload.get("temporal_mode"),
                    "filters": value.payload.get("filters", {}),
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in value.citations
                    ],
                }
            )
        traced[query_id] = rows
    return traced


def _missing_answer_ids(queries, answers_dir: Path) -> list[str]:
    answer_ids = {
        answer.query_id
        for path in answers_dir.glob("*.json")
        for answer in [
            AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))
        ]
        if answer.valid
    }
    return [query.query_id for query in queries if query.query_id not in answer_ids]


def command_experiment(config: AppConfig, args) -> dict:
    paths = _paths(config)
    queries = load_queries(paths["queries"])
    judgments = load_judgments(paths["judgments"])
    records, encoder, retriever = _load_index(config)
    if config.retrieval.collection not in {
        item.name for item in retriever.client.get_collections().collections
    }:
        retriever.build()
    lexical = retriever.lexical
    validation = [query for query in queries if query.partition == "validation"]
    fine_records = fine_encoder = fine_retriever = None
    if (paths["model"] / "modules.json").is_file():
        fine_records, fine_encoder, fine_retriever = _load_index(
            config, model_path=paths["model"]
        )
        if config.retrieval.collection not in {
            item.name for item in fine_retriever.client.get_collections().collections
        }:
            fine_retriever.build()
    trials = []
    tuning_systems = [("pretrained", encoder, retriever)]
    if fine_encoder is not None:
        tuning_systems.append(("fine_tuned", fine_encoder, fine_retriever))
    for model_variant, tuning_encoder, tuning_retriever in tuning_systems:
        original_config = tuning_retriever.config
        validation_vectors = tuning_encoder.encode_queries(
            [query.query for query in validation]
        )
        for relevance_floor in (0.10, 0.15, 0.20):
            for temporal_weight in (0.0, 0.10, 0.15, 0.20):
                tuning_retriever.config = original_config.model_copy(
                    update={
                        "relevance_floor": relevance_floor,
                        "temporal_weight": temporal_weight,
                    }
                )
                rankings = {}
                for query, vector in zip(
                    validation, validation_vectors, strict=True
                ):
                    rankings[query.query_id] = tuning_retriever.search(
                        query.query,
                        vector,
                        temporal_mode=query.temporal_mode,
                        as_of=query.as_of,
                        filters={"partition": query.partition},
                        limit=50,
                    )
                metrics = retrieval_metrics(rankings, judgments, validation)
                trials.append(
                    {
                        "model_variant": model_variant,
                        "relevance_floor": relevance_floor,
                        "temporal_weight": temporal_weight,
                        "metrics": metrics["aggregate"],
                    }
                )
        tuning_retriever.config = original_config
    best = max(
        trials,
        key=lambda trial: (
            trial["metrics"]["ndcg@10"],
            trial["metrics"]["mrr@10"],
            -trial["temporal_weight"],
        ),
    )
    retriever.config = retriever.config.model_copy(
        update={
            "relevance_floor": best["relevance_floor"],
            "temporal_weight": best["temporal_weight"],
        }
    )

    marker = config.derived_dir / "evaluation" / ".frozen-evaluation-complete"
    if args.run_frozen_evaluation:
        if marker.exists():
            raise RuntimeError(
                "frozen pilot evaluation was already run; preserve it unchanged"
            )
    partitions = ["validation"]
    experiments = {"validation": {}}
    traces = {"validation": {}}
    validation_rankings = _experiment_rankings(
        validation, encoder, retriever, lexical, records
    )
    for name, values in validation_rankings.items():
        experiments["validation"][name] = retrieval_metrics(
            values, judgments, validation
        )
        traces["validation"][name] = _trace_rankings(values, name)

    if fine_retriever is not None:
        fine_retriever.config = fine_retriever.config.model_copy(
            update={
                "relevance_floor": best["relevance_floor"],
                "temporal_weight": best["temporal_weight"],
            }
        )
        fine_rankings = _experiment_rankings(
            validation,
            fine_encoder,
            fine_retriever,
            fine_retriever.lexical,
            fine_records,
        )
        for name, source_name in (
            ("fine_tuned_dense", "dense"),
            ("fine_tuned_hybrid", "hybrid_temporal"),
        ):
            values = fine_rankings[source_name]
            experiments["validation"][name] = retrieval_metrics(
                values, judgments, validation
            )
            traces["validation"][name] = _trace_rankings(values, name)

    validation_aggregates = {
        name: metrics["aggregate"]
        for name, metrics in experiments["validation"].items()
    }
    best_validation_variant = max(
        validation_aggregates,
        key=lambda name: (
            validation_aggregates[name]["ndcg@10"],
            validation_aggregates[name]["mrr@10"],
        ),
    )
    variant_names = {
        "bm25": "bm25",
        "dense": "pretrained_dense",
        "hybrid": "pretrained_hybrid",
        "hybrid_temporal": "pretrained_hybrid_temporal",
        "fine_tuned_dense": "fine_tuned_dense",
        "fine_tuned_hybrid": "fine_tuned_hybrid",
    }
    frozen = {
        "selected_on": "validation",
        "model_variant": variant_names[best_validation_variant],
        "relevance_floor": best["relevance_floor"],
        "temporal_weight": best["temporal_weight"],
        "selection_metric": "ndcg@10",
        "validation_metrics": validation_aggregates[best_validation_variant],
        "config_hash": canonical_hash(config.model_dump(mode="json")),
    }
    frozen_path = config.derived_dir / "evaluation" / "frozen-config.json"
    if args.run_frozen_evaluation:
        if not frozen_path.is_file():
            raise RuntimeError(
                "validation configuration has not been frozen; run experiment first"
            )
        prior_frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if prior_frozen != frozen:
            raise RuntimeError(
                "validation-selected configuration changed after freezing; "
                "refusing the one-shot evaluation"
            )
        evaluation_queries = [
            query for query in queries if query.partition == "evaluation"
        ]
        missing_answers = _missing_answer_ids(
            evaluation_queries,
            config.derived_dir / "answers",
        )
        if missing_answers:
            raise RuntimeError(
                "cannot seal evaluation without answers for every frozen query; "
                f"missing={len(missing_answers)}"
            )
        partitions.append("evaluation")
        traces["evaluation"] = {}
        evaluation_rankings = _experiment_rankings(
            evaluation_queries, encoder, retriever, lexical, records
        )
        experiments["evaluation"] = {}
        for name, values in evaluation_rankings.items():
            experiments["evaluation"][name] = retrieval_metrics(
                values, judgments, evaluation_queries
            )
            traces["evaluation"][name] = _trace_rankings(values, name)
        if fine_retriever is not None:
            fine_evaluation_rankings = _experiment_rankings(
                evaluation_queries,
                fine_encoder,
                fine_retriever,
                fine_retriever.lexical,
                fine_records,
            )
            for name, source_name in (
                ("fine_tuned_dense", "dense"),
                ("fine_tuned_hybrid", "hybrid_temporal"),
            ):
                values = fine_evaluation_rankings[source_name]
                experiments["evaluation"][name] = retrieval_metrics(
                    values, judgments, evaluation_queries
                )
                traces["evaluation"][name] = _trace_rankings(values, name)
    else:
        write_json_atomic(frozen_path, frozen)
    write_json_atomic(
        config.derived_dir / "evaluation" / "retrieval-traces.json",
        traces,
    )

    query_summary = benchmark_summary(queries, [], judgments)
    benchmark_summaries = {}
    for name in ("summary", "judgment-summary"):
        path = config.derived_dir / "benchmark" / f"{name}.json"
        if path.is_file():
            benchmark_summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    answer_quality = {}
    answers_dir = config.derived_dir / "answers"
    if answers_dir.is_dir():
        answers = [
            AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in answers_dir.glob("*.json")
        ]
        for partition in partitions:
            selected = [query for query in queries if query.partition == partition]
            answer_quality[partition] = answer_metrics(answers, selected, judgments)
    assessment_path = config.derived_dir / "benchmark" / "critic-assessments.json"
    assessments = (
        json.loads(assessment_path.read_text(encoding="utf-8"))
        if assessment_path.is_file()
        else []
    )
    rejection_issues = Counter()
    for assessment in assessments:
        if assessment.get("accepted"):
            continue
        if assessment.get("deterministic_errors"):
            rejection_issues["deterministic"] += 1
        if assessment.get("critic_failures"):
            rejection_issues["critic_failure"] += 1
        for score in assessment.get("scores", []):
            for field in (
                "ambiguity_issue",
                "leakage_issue",
                "temporal_issue",
            ):
                if score.get(field):
                    rejection_issues[field] += 1
            if not score.get("accepted"):
                rejection_issues["critic_rejected"] += 1
    training_report_path = paths["model"] / "training-report.json"
    report = {
        "dataset_scope": "entireio-cli only",
        "evidence_corpus": json.loads(paths["quality"].read_text(encoding="utf-8")),
        "partition_coverage": json.loads(
            paths["partition"].read_text(encoding="utf-8")
        ),
        "indexed_evidence": len(records),
        "evidence_types": dict(
            Counter(record.evidence_type.value for record in records)
        ),
        "queries": query_summary,
        "benchmark_runs": benchmark_summaries,
        "generation_acceptance": {
            "assessed": len(assessments),
            "accepted": sum(bool(item.get("accepted")) for item in assessments),
            "rejected": sum(not item.get("accepted") for item in assessments),
            "rejection_issue_counts": dict(rejection_issues),
        },
        "api_usage": _cached_api_usage(config.derived_dir / "openrouter-cache"),
        "resource_usage": (
            json.loads(training_report_path.read_text(encoding="utf-8"))
            if training_report_path.is_file()
            else {"training": "not yet run"}
        ),
        "tuning_trials": trials,
        "frozen_config": frozen,
        "experiments": experiments,
        "validation_findings": {
            "best_variant_by_ndcg@10": best_validation_variant,
            "best_ndcg@10": (
                validation_aggregates[best_validation_variant]["ndcg@10"]
                if best_validation_variant
                else None
            ),
            "fine_tuned_dense_delta_vs_pretrained": (
                validation_aggregates["fine_tuned_dense"]["ndcg@10"]
                - validation_aggregates["dense"]["ndcg@10"]
                if {"fine_tuned_dense", "dense"} <= validation_aggregates.keys()
                else None
            ),
            "fine_tuned_hybrid_delta_vs_pretrained": (
                validation_aggregates["fine_tuned_hybrid"]["ndcg@10"]
                - validation_aggregates["hybrid_temporal"]["ndcg@10"]
                if {"fine_tuned_hybrid", "hybrid_temporal"}
                <= validation_aggregates.keys()
                else None
            ),
            "temporal_weight_selected": best["temporal_weight"],
        },
        "answer_quality": answer_quality,
        "artifact_hashes": {
            str(path.relative_to(config.derived_dir)): sha256_file(path)
            for path in (
                paths["evidence"],
                paths["quality"],
                paths["partition"],
                paths["queries"],
                paths["judgments"],
                paths["vectors"],
                paths["vectors_fine"],
                paths["model"] / "model.safetensors",
                paths["model"] / "training-report.json",
                frozen_path,
                config.derived_dir / "evaluation" / "retrieval-traces.json",
            )
            if path.is_file()
        },
        "limitations": [
            "Silver labels are generated and judged by the configured language model.",
            "The source is a bounded entireio-cli dataset, not a complete repository snapshot.",
            "The compact CPU pilot omits low-value tool traffic from the vector index.",
            "Local embedded Qdrant is appropriate for this pilot; payload indexes require Qdrant server mode to accelerate larger corpora.",
        ],
    }
    write_json_atomic(config.derived_dir / "evaluation" / "pilot-report.json", report)
    if args.run_frozen_evaluation:
        marker.write_text(canonical_hash(report) + "\n", encoding="utf-8")
    _record_manifest(
        config,
        "experiment-frozen" if args.run_frozen_evaluation else "experiment-validation",
        input_paths=[
            paths["evidence"],
            paths["quality"],
            paths["partition"],
            paths["queries"],
            paths["judgments"],
            paths["vectors"],
            paths["vectors_fine"],
        ],
        artifact_paths=[
            config.derived_dir / "evaluation" / "frozen-config.json",
            config.derived_dir / "evaluation" / "pilot-report.json",
            config.derived_dir / "evaluation" / "retrieval-traces.json",
            *([marker] if marker.is_file() else []),
        ],
        model_settings={
            **config.embedding.model_dump(mode="json"),
            "selected_variant": frozen["model_variant"],
        },
        counts={
            "validation_queries": len(validation),
            "evaluation_queries": sum(
                query.partition == "evaluation" for query in queries
            )
            if args.run_frozen_evaluation
            else 0,
        },
    )
    print(json.dumps({"frozen_config": frozen, "partitions": partitions}, indent=2))
    return report


def command_calibrate(config: AppConfig, args) -> dict:
    paths = _paths(config)
    traces_path = config.derived_dir / "evaluation" / "retrieval-traces.json"
    if not traces_path.is_file():
        raise RuntimeError("retrieval traces are missing; run experiment first")
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    if "validation" not in traces:
        raise RuntimeError("validation traces are required for calibration")
    queries = load_queries(paths["queries"])
    judgments = load_judgments(paths["judgments"])
    validation_queries = [
        query for query in queries if query.partition == "validation"
    ]
    evaluation_queries = [
        query for query in queries if query.partition == "evaluation"
    ]
    artifact = fit_calibration(
        traces["validation"],
        validation_queries,
        judgments,
        evaluation_traces=traces.get("evaluation"),
        evaluation_queries=(
            evaluation_queries if "evaluation" in traces else None
        ),
    )
    artifact["retrieval_impact"] = {}
    for partition, selected_queries in (
        ("validation", validation_queries),
        ("evaluation", evaluation_queries),
    ):
        if partition not in traces:
            continue
        original = {
            query_id: [row["evidence_id"] for row in rows]
            for query_id, rows in traces[partition]["dense"].items()
        }
        calibrated = {
            query_id: [
                row["evidence_id"]
                for row in rows
                if row.get("dense_score") is not None
                and row["dense_score"]
                >= artifact["relevance"]["raw_score_threshold"]
            ]
            for query_id, rows in traces[partition]["dense"].items()
        }
        artifact["retrieval_impact"][partition] = {
            "before": retrieval_metrics(
                original, judgments, selected_queries
            )["aggregate"],
            "after": retrieval_metrics(
                calibrated, judgments, selected_queries
            )["aggregate"],
        }
    validation_impact = artifact["retrieval_impact"]["validation"]
    artifact["relevance"]["enabled_for_filtering"] = bool(
        validation_impact["after"]["ndcg@10"]
        >= validation_impact["before"]["ndcg@10"]
        and validation_impact["after"]["recall@20"]
        >= validation_impact["before"]["recall@20"] * 0.95
    )
    artifact["usage"] = {
        "fit_inputs": ["validation"],
        "evaluation_is_holdout_reporting_only": "evaluation" in traces,
        "sealed_frozen_configuration_unchanged": True,
    }
    artifact["limitations"] = [
        "Only five validation questions are labelled unsupported_by_dataset.",
        "Absolute cosine-score calibration is model- and corpus-specific.",
        "Evaluation metrics are diagnostic and were not used to select thresholds.",
        "Lexical top-five matches bypass the dense threshold for exact-term recall.",
    ]
    write_json_atomic(paths["calibration"], artifact)
    _record_manifest(
        config,
        "calibrate",
        input_paths=[
            paths["queries"],
            paths["judgments"],
            traces_path,
            config.derived_dir / "evaluation" / "frozen-config.json",
            config.derived_dir / "evaluation" / ".frozen-evaluation-complete",
        ],
        artifact_paths=[paths["calibration"]],
        model_settings={
            "model_variant": artifact["model_variant"],
            "version": artifact["version"],
        },
        counts={
            "validation_queries": len(validation_queries),
            "evaluation_queries": len(evaluation_queries)
            if "evaluation" in traces
            else 0,
        },
    )
    print(
        json.dumps(
            {
                "version": artifact["version"],
                "raw_score_threshold": artifact["relevance"][
                    "raw_score_threshold"
                ],
                "relevance_validation_oof": artifact["relevance"][
                    "validation_oof_metrics"
                ],
                "relevance_filtering_enabled": artifact["relevance"][
                    "enabled_for_filtering"
                ],
                "answerability_validation": artifact["answerability"][
                    "validation_oof_metrics"
                ],
                "answerability_abstention_enabled": artifact["answerability"][
                    "enabled_for_abstention"
                ],
                "retrieval_impact": artifact["retrieval_impact"],
                "holdout_evaluation": artifact.get("holdout_evaluation"),
            },
            indent=2,
        )
    )
    return artifact


def command_audit(config: AppConfig, args) -> dict:
    paths = _paths(config)
    records = load_evidence(paths["evidence"])
    allowed_tables = {
        "sessions.parquet",
        "session_logs.parquet",
        "conversations.parquet",
        "checkpoints.parquet",
        "commits.parquet",
    }
    violations = []
    source_counts = Counter()
    for record in records:
        if not record.source_refs:
            violations.append(f"{record.evidence_id}: no source reference")
        for ref in record.source_refs:
            source_counts[ref.source_file] += 1
            allowed = (
                ref.source_file in allowed_tables
                or ref.source_file.startswith("transcripts/")
            )
            if not allowed:
                violations.append(
                    f"{record.evidence_id}: disallowed source {ref.source_file}"
                )
    expected_data_dir = Path(__file__).resolve().parents[2]
    if config.data_dir != expected_data_dir:
        violations.append(
            f"configured data directory is outside entireio-cli: {config.data_dir}"
        )
    manifest_violations = []
    manifest_dir = config.derived_dir / "manifests"
    for manifest_path in manifest_dir.glob("*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for input_path in manifest.get("input_hashes", {}):
            resolved = Path(input_path).resolve()
            if resolved != config.data_dir and config.data_dir not in resolved.parents:
                manifest_violations.append(
                    f"{manifest_path.name}: external input {resolved}"
                )
    violations.extend(manifest_violations)
    report = {
        "passed": not violations,
        "scope": str(config.data_dir),
        "evidence_records": len(records),
        "allowed_source_files": sorted(allowed_tables),
        "observed_source_counts": dict(sorted(source_counts.items())),
        "manifest_count": len(list(manifest_dir.glob("*.json"))),
        "violations": violations,
    }
    write_json_atomic(config.derived_dir / "evaluation" / "dataset-boundary-audit.json", report)
    if violations:
        raise AssertionError(
            "dataset-boundary audit failed: " + "; ".join(violations[:10])
        )
    print(json.dumps(report, indent=2))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="entireio-retrieval")
    result.add_argument("--config", type=Path, default=_default_config())
    sub = result.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--fix-key-permissions", action="store_true")

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--full", action="store_true")

    generate = sub.add_parser("generate")
    generate.add_argument("--per-group", type=int, default=5)
    generate.add_argument("--max-groups", type=int)
    generate.add_argument("--double-evaluation", action="store_true")

    judge = sub.add_parser("judge")
    judge.add_argument("--limit", type=int, default=100)
    judge.add_argument("--rebuild-pools", action="store_true")

    index = sub.add_parser("index")
    index.add_argument("--recreate", action="store_true")
    index.add_argument("--fine-tuned", action="store_true")

    query = sub.add_parser("query")
    query.add_argument("query")
    query.add_argument("--temporal-mode", choices=[item.value for item in TemporalMode])
    query.add_argument("--fine-tuned", action="store_true")
    query.add_argument("--no-calibration", action="store_true")

    answer = sub.add_parser("answer")
    answer.add_argument("query")
    answer.add_argument("--temporal-mode", choices=[item.value for item in TemporalMode])
    answer.add_argument("--fine-tuned", action="store_true")
    answer.add_argument("--no-calibration", action="store_true")

    answer_benchmark = sub.add_parser("answer-benchmark")
    answer_benchmark.add_argument(
        "--partition", choices=["validation", "evaluation"], required=True
    )
    answer_benchmark.add_argument("--fine-tuned", action="store_true")

    train = sub.add_parser("train")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--partition", choices=["validation", "evaluation"], default="validation")
    evaluate.add_argument("--fine-tuned", action="store_true")

    experiment = sub.add_parser("experiment")
    experiment.add_argument("--run-frozen-evaluation", action="store_true")

    sub.add_parser("calibrate")
    sub.add_parser("audit")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    handlers = {
        "validate": command_validate,
        "prepare": command_prepare,
        "generate": command_generate,
        "judge": command_judge,
        "index": command_index,
        "query": command_query,
        "answer": command_answer,
        "answer-benchmark": command_answer_benchmark,
        "train": command_train,
        "evaluate": command_evaluate,
        "experiment": command_experiment,
        "calibrate": command_calibrate,
        "audit": command_audit,
    }
    handlers[args.command](config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
