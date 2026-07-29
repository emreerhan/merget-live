from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import (
    Answerability,
    CriticResponse,
    EvidenceRecord,
    GeneratedQueryBatch,
    JudgmentBatch,
    QueryRecord,
    RelevanceJudgment,
    TemporalMode,
)
from .openrouter import OpenRouterClient
from .prompts import (
    CRITIC_PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    QUERY_PROMPT_VERSION,
    QUERY_SYSTEM_PROMPT,
    UNSUPPORTED_CRITIC_PROMPT_VERSION,
    UNSUPPORTED_CRITIC_SYSTEM_PROMPT,
)
from .provenance import canonical_hash, stable_id, write_json_atomic, write_jsonl_atomic
from .security import contains_sensitive_text

if TYPE_CHECKING:
    from .embeddings import DenseEncoder
    from .lexical import BM25SparseIndex

META_RE = re.compile(r"\b(?:provided|supplied|above)\s+(?:context|evidence|packet)\b", re.I)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.I)
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", re.I)


def evidence_packet(records: list[EvidenceRecord], max_chars: int = 24000) -> dict[str, Any]:
    remaining = max_chars
    items = []
    for record in records:
        text = record.text[: min(remaining, 6000)]
        if not text:
            continue
        items.append(
            {
                "evidence_id": record.evidence_id,
                "type": record.evidence_type.value,
                "title": record.title,
                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                "text": text,
                "files": record.files[:30],
                "session_ids": record.session_ids,
                "commit_shas": record.commit_shas,
            }
        )
        remaining -= len(text)
        if remaining <= 0:
            break
    return {"repository": "entireio/cli", "evidence": items}


def group_generation_sources(records: list[EvidenceRecord]) -> list[list[EvidenceRecord]]:
    topics = [record for record in records if record.evidence_type.value == "topic"]
    by_id = {record.evidence_id: record for record in records}
    groups: list[list[EvidenceRecord]] = []
    used_sessions: set[str] = set()
    for topic in topics:
        members = [by_id[parent] for parent in topic.parent_ids if parent in by_id][:12]
        groups.append([topic] + members)
        used_sessions.update(topic.session_ids)
    for record in records:
        if record.evidence_type.value == "session" and not set(record.session_ids) <= used_sessions:
            groups.append([record] + [by_id[parent] for parent in record.parent_ids if parent in by_id][:8])
    return groups


class QueryBenchmarkBuilder:
    def __init__(
        self,
        client: OpenRouterClient,
        records: list[EvidenceRecord],
        artifact_dir: Path,
        *,
        seed: int = 20260729,
    ):
        self.client = client
        self.records = records
        self.by_id = {record.evidence_id: record for record in records}
        self.artifact_dir = artifact_dir
        self.seed = seed

    def generate(self, per_group: int = 5, max_groups: int | None = None) -> list[QueryRecord]:
        groups = group_generation_sources(self.records)
        if max_groups is not None:
            groups = groups[:max_groups]
        requests = []
        for index, group in enumerate(groups):
            partition = max(
                (record.partition or "train" for record in group),
                key=("train", "validation", "evaluation").index,
            )
            user_payload = {
                "prompt_version": QUERY_PROMPT_VERSION,
                "requested_count": per_group,
                "partition": partition,
                "requirements": {
                    "vary_scope_and_style": True,
                    "unsupported_fraction": 0.1,
                    "do_not_copy_source": True,
                },
                **evidence_packet(group),
            }
            requests.append(
                {
                    "messages": [
                        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_payload, default=str)},
                    ],
                    "response_model": GeneratedQueryBatch,
                    "purpose": f"query-generation-{index:05d}",
                }
            )
        results = asyncio.run(self.client.complete_many(requests))
        accepted: list[QueryRecord] = []
        failures = []
        for group_index, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append({"group": group_index, "error": type(result).__name__, "message": str(result)})
                continue
            for query_index, query in enumerate(result.queries):
                group = groups[group_index]
                partition = max(
                    (record.partition or "train" for record in group),
                    key=("train", "validation", "evaluation").index,
                )
                query_id = stable_id(
                    "query", QUERY_PROMPT_VERSION, group_index, query_index, query.query
                )
                provenance = {
                    **query.provenance,
                    "generator_prompt": QUERY_PROMPT_VERSION,
                    "model": self.client.config.model,
                    "reasoning_effort": self.client.config.reasoning_effort,
                    "source_group": group_index,
                }
                accepted.append(
                    query.model_copy(
                        update={
                            "query_id": query_id,
                            "partition": partition,
                            "provenance": provenance,
                        }
                    )
                )
        write_json_atomic(self.artifact_dir / "generation-failures.json", failures)
        write_jsonl_atomic(self.artifact_dir / "generated-queries.jsonl", accepted)
        return accepted

    def criticize(self, queries: list[QueryRecord], *, double_evaluation: bool = False) -> tuple[list[QueryRecord], list[dict[str, Any]]]:
        assessments: list[dict[str, Any]] = []
        accepted: list[QueryRecord] = []
        jobs: list[tuple[str, int]] = []
        requests: list[dict[str, Any]] = []
        for query in queries:
            source_ids = list(dict.fromkeys(query.generation_source_ids + query.primary_positive_ids + query.supporting_positive_ids))
            source_records = [self.by_id[item] for item in source_ids if item in self.by_id]
            passes = 2 if double_evaluation and query.partition == "evaluation" else 1
            unsupported = query.answerability == Answerability.UNSUPPORTED
            critic_version = (
                UNSUPPORTED_CRITIC_PROMPT_VERSION
                if unsupported
                else CRITIC_PROMPT_VERSION
            )
            critic_prompt = (
                UNSUPPORTED_CRITIC_SYSTEM_PROMPT
                if unsupported
                else CRITIC_SYSTEM_PROMPT
            )
            for pass_index in range(passes):
                shuffled = list(source_records)
                random.Random(f"{self.seed}:{query.query_id}:{pass_index}").shuffle(shuffled)
                payload = {
                    "prompt_version": critic_version,
                    "candidate": query.model_dump(mode="json"),
                    **evidence_packet(shuffled),
                }
                jobs.append((query.query_id, pass_index))
                requests.append(
                    {
                        "messages": [
                            {"role": "system", "content": critic_prompt},
                            {"role": "user", "content": json.dumps(payload, default=str)},
                        ],
                        "response_model": CriticResponse,
                        "purpose": f"query-critic-{query.query_id}-{pass_index}",
                    }
                )
        results = asyncio.run(self.client.complete_many(requests))
        scores_by_query: dict[str, list[tuple[int, Any]]] = defaultdict(list)
        failures_by_query: dict[str, list[str]] = defaultdict(list)
        for (query_id, pass_index), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures_by_query[query_id].append(
                    f"{type(result).__name__}: {result}"
                )
            else:
                scores_by_query[query_id].append((pass_index, result.score))

        for query in queries:
            passes = 2 if double_evaluation and query.partition == "evaluation" else 1
            scores = [
                score
                for _, score in sorted(scores_by_query.get(query.query_id, []))
            ]
            deterministic_errors = validate_query(query, self.by_id)
            pass_threshold = len(scores) == passes and all(
                score.accepted
                and score.groundedness == 5
                and score.answerability >= 4
                and score.evidence_sufficiency >= 4
                and score.naturalness >= 3
                and not score.ambiguity_issue
                and not score.leakage_issue
                and not score.temporal_issue
                for score in scores
            )
            is_accepted = pass_threshold and not deterministic_errors
            assessment = {
                "query_id": query.query_id,
                "accepted": is_accepted,
                "scores": [score.model_dump(mode="json") for score in scores],
                "deterministic_errors": deterministic_errors,
                "critic_failures": failures_by_query.get(query.query_id, []),
            }
            assessments.append(assessment)
            if is_accepted:
                accepted.append(query)
        accepted = semantic_deduplicate(accepted)
        write_jsonl_atomic(self.artifact_dir / "accepted-queries.jsonl", accepted)
        write_json_atomic(self.artifact_dir / "critic-assessments.json", assessments)
        return accepted, assessments

    def judge_candidates(
        self,
        queries: list[QueryRecord],
        candidate_ids: dict[str, list[str]],
    ) -> list[RelevanceJudgment]:
        output: list[RelevanceJudgment] = []
        requests: list[dict[str, Any]] = []
        requested_queries: list[QueryRecord] = []
        for query in queries:
            candidates = [self.by_id[item] for item in candidate_ids.get(query.query_id, []) if item in self.by_id]
            random.Random(f"{self.seed}:{query.query_id}:judge").shuffle(candidates)
            payload = {
                "prompt_version": JUDGE_PROMPT_VERSION,
                "query": query.model_dump(mode="json"),
                "candidates": evidence_packet(candidates, max_chars=50000)["evidence"],
            }
            requested_queries.append(query)
            requests.append(
                {
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, default=str)},
                    ],
                    "response_model": JudgmentBatch,
                    "purpose": f"relevance-judge-{query.query_id}",
                }
            )
        results = asyncio.run(self.client.complete_many(requests))
        failures = []
        for query, result in zip(requested_queries, results):
            if isinstance(result, Exception):
                failures.append(
                    {
                        "query_id": query.query_id,
                        "error": type(result).__name__,
                        "message": str(result),
                    }
                )
                continue
            candidate_set = set(candidate_ids.get(query.query_id, []))
            expected_claim_ids = {
                claim.claim_id for claim in query.expected_claims
            }
            for judgment in result.judgments:
                if (
                    judgment.evidence_id not in candidate_set
                    or judgment.evidence_id not in self.by_id
                ):
                    failures.append(
                        {
                            "query_id": query.query_id,
                            "error": "invalid_evidence_id",
                            "message": judgment.evidence_id,
                        }
                    )
                    continue
                record = self.by_id[judgment.evidence_id]
                valid_quotes = [
                    quote
                    for quote in judgment.supporting_quotes
                    if quote and quote in record.text
                ]
                valid_claims = [
                    claim_id
                    for claim_id in judgment.supported_claim_ids
                    if claim_id in expected_claim_ids
                ]
                grade = judgment.grade
                if (
                    query.answerability == Answerability.UNSUPPORTED
                    or (grade > 0 and (not valid_quotes or not valid_claims))
                ):
                    grade, valid_quotes, valid_claims = 0, [], []
                if grade == 0:
                    valid_quotes, valid_claims = [], []
                output.append(
                    judgment.model_copy(
                        update={
                            "query_id": query.query_id,
                            "grade": grade,
                            "supporting_quotes": valid_quotes,
                            "supported_claim_ids": valid_claims,
                            "provenance": {
                                **judgment.provenance,
                                "prompt_version": JUDGE_PROMPT_VERSION,
                                "model": self.client.config.model,
                                "deterministically_normalized": True,
                            }
                        }
                    )
                )
        output = _deduplicate_and_anchor_judgments(output, queries, self.by_id)
        write_jsonl_atomic(self.artifact_dir / "relevance-judgments.jsonl", output)
        write_json_atomic(self.artifact_dir / "relevance-failures.json", failures)
        return output


def _deduplicate_and_anchor_judgments(
    judgments: list[RelevanceJudgment],
    queries: list[QueryRecord],
    evidence: dict[str, EvidenceRecord],
) -> list[RelevanceJudgment]:
    query_by_id = {query.query_id: query for query in queries}
    best: dict[tuple[str, str], RelevanceJudgment] = {}
    for judgment in judgments:
        key = (judgment.query_id, judgment.evidence_id)
        existing = best.get(key)
        if existing is None or judgment.grade > existing.grade:
            best[key] = judgment
    for query in queries:
        if query.answerability == Answerability.UNSUPPORTED:
            continue
        claim_ids_by_evidence: dict[str, list[str]] = defaultdict(list)
        for claim in query.expected_claims:
            for evidence_id in claim.supporting_evidence_ids:
                claim_ids_by_evidence[evidence_id].append(claim.claim_id)
        for grade, evidence_ids in (
            (3, query.primary_positive_ids),
            (2, query.supporting_positive_ids),
        ):
            for evidence_id in evidence_ids:
                record = evidence.get(evidence_id)
                if record is None:
                    continue
                key = (query.query_id, evidence_id)
                claims = list(dict.fromkeys(claim_ids_by_evidence.get(evidence_id, [])))
                if not claims:
                    continue
                anchor = RelevanceJudgment(
                    query_id=query.query_id,
                    evidence_id=evidence_id,
                    grade=grade,
                    supported_claim_ids=claims,
                    supporting_quotes=[record.text[: min(240, len(record.text))]],
                    provenance={
                        "source": "critic-validated-generator-positive",
                        "deterministically_anchored": True,
                    },
                )
                existing = best.get(key)
                if existing is None or existing.grade < grade:
                    best[key] = anchor
    return sorted(
        best.values(),
        key=lambda judgment: (
            query_by_id.get(judgment.query_id).partition
            if judgment.query_id in query_by_id
            else "unknown",
            judgment.query_id,
            judgment.evidence_id,
        ),
    )


def validate_query(query: QueryRecord, evidence: dict[str, EvidenceRecord]) -> list[str]:
    errors: list[str] = []
    if META_RE.search(query.query):
        errors.append("query contains evidence-packet meta language")
    if UUID_RE.search(query.query) or SHA_RE.search(query.query):
        errors.append("query leaks dataset-native identifiers")
    if contains_sensitive_text(query.query):
        errors.append("query contains sensitive text")
    source_ids = set(query.generation_source_ids)
    positive_ids = set(query.primary_positive_ids + query.supporting_positive_ids)
    claim_ids = {
        evidence_id
        for claim in query.expected_claims
        for evidence_id in claim.supporting_evidence_ids
    }
    missing = (source_ids | positive_ids | claim_ids) - evidence.keys()
    if missing:
        errors.append(f"missing evidence IDs: {sorted(missing)}")
    cross_partition = {
        evidence_id
        for evidence_id in source_ids | positive_ids | claim_ids
        if evidence_id in evidence
        and evidence[evidence_id].partition != query.partition
    }
    if cross_partition:
        errors.append(
            f"evidence crosses query partition: {sorted(cross_partition)}"
        )
    if query.answerability == Answerability.UNSUPPORTED:
        if query.expected_claims or positive_ids:
            errors.append("unsupported query has claims or positive evidence")
    elif not query.expected_claims or not positive_ids:
        errors.append("supported/partial query lacks claims or positive evidence")
    for claim in query.expected_claims:
        for evidence_id, quote in claim.evidence_quotes.items():
            record = evidence.get(evidence_id)
            if record is None or quote not in record.text:
                errors.append(f"claim {claim.claim_id} quote does not resolve in {evidence_id}")
    return errors


def semantic_deduplicate(
    queries: list[QueryRecord], threshold: float = 0.92
) -> list[QueryRecord]:
    if len(queries) < 2:
        return queries
    normalized = [" ".join(query.query.lower().split()) for query in queries]
    exact_seen: set[str] = set()
    exact_unique: list[QueryRecord] = []
    for text, query in zip(normalized, queries):
        if text not in exact_seen:
            exact_seen.add(text)
            exact_unique.append(query)
    if len(exact_unique) < 2:
        return exact_unique
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
        [query.query for query in exact_unique]
    )
    similarities = cosine_similarity(matrix)
    keep: list[int] = []
    for index in range(len(exact_unique)):
        if all(similarities[index, prior] < threshold for prior in keep):
            keep.append(index)
    return [exact_unique[index] for index in keep]


def pool_candidates(*rankings: dict[str, list[str]], limit: int = 100) -> dict[str, list[str]]:
    pooled: dict[str, list[str]] = defaultdict(list)
    for ranking in rankings:
        for query_id, evidence_ids in ranking.items():
            for evidence_id in evidence_ids:
                if evidence_id not in pooled[query_id]:
                    pooled[query_id].append(evidence_id)
                    if len(pooled[query_id]) >= limit:
                        break
    return dict(pooled)


def build_candidate_pool(
    queries: list[QueryRecord],
    records: list[EvidenceRecord],
    encoder: "DenseEncoder",
    vectors: dict[str, np.ndarray],
    lexical: "BM25SparseIndex",
    *,
    per_source: int = 40,
    limit: int = 100,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Pool candidates without conditioning inclusion on baseline success."""
    by_id = {record.evidence_id: record for record in records}
    partition_records: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        partition_records[record.partition or "train"].append(record)
    partition_dense: dict[str, tuple[list[str], np.ndarray]] = {}
    for partition, partition_items in partition_records.items():
        ids = [
            record.evidence_id
            for record in partition_items
            if record.evidence_id in vectors
        ]
        partition_dense[partition] = (
            ids,
            np.vstack([vectors[evidence_id] for evidence_id in ids]),
        )
    query_vectors = encoder.encode_queries([query.query for query in queries])
    pooled: dict[str, list[str]] = {}
    traces: dict[str, dict[str, list[str]]] = {}
    for query, query_vector in zip(queries, query_vectors):
        allowed = partition_records[query.partition]
        allowed_ids = {record.evidence_id for record in allowed}
        dense_ids, dense_matrix = partition_dense[query.partition]
        dense_order = np.argsort(-(dense_matrix @ query_vector))[:per_source]
        dense = [dense_ids[index] for index in dense_order]
        bm25 = [
            evidence_id
            for evidence_id, _ in lexical.rank(query.query, limit=per_source * 3)
            if evidence_id in allowed_ids
        ][:per_source]

        seeds = [
            by_id[evidence_id]
            for evidence_id in dict.fromkeys(
                query.generation_source_ids
                + query.primary_positive_ids
                + query.supporting_positive_ids
            )
            if evidence_id in by_id
        ]
        sessions = {value for record in seeds for value in record.session_ids}
        checkpoints = {value for record in seeds for value in record.checkpoint_pks}
        files = {value for record in seeds for value in record.files}
        metadata_scores = []
        for record in allowed:
            score = (
                3 * len(sessions.intersection(record.session_ids))
                + 2 * len(checkpoints.intersection(record.checkpoint_pks))
                + len(files.intersection(record.files))
            )
            if score:
                metadata_scores.append((record.evidence_id, score))
        metadata_scores.sort(key=lambda item: (-item[1], item[0]))
        metadata = [item[0] for item in metadata_scores[:per_source]]

        chronological = sorted(
            (record for record in allowed if record.timestamp),
            key=lambda record: (record.timestamp, record.evidence_id),
        )
        if query.temporal_mode in {TemporalMode.CURRENT, TemporalMode.LATEST}:
            chronological.reverse()
        elif query.temporal_mode == TemporalMode.NEUTRAL:
            # Interleave old, middle, and recent records for temporal diversity.
            chronological = [
                item
                for pair in zip(chronological, reversed(chronological))
                for item in pair
            ]
        temporal = list(
            dict.fromkeys(record.evidence_id for record in chronological)
        )[:per_source]

        trace = {
            "bm25": bm25,
            "pretrained_dense": dense,
            "metadata_neighbor": metadata,
            "temporally_diverse": temporal,
        }
        traces[query.query_id] = trace
        pooled[query.query_id] = pool_candidates(
            *(
                {query.query_id: evidence_ids}
                for evidence_ids in trace.values()
            ),
            limit=limit,
        ).get(query.query_id, [])
    return pooled, traces


def benchmark_summary(
    queries: list[QueryRecord],
    assessments: list[dict[str, Any]],
    judgments: list[RelevanceJudgment] | None = None,
) -> dict[str, Any]:
    return {
        "accepted_queries": len(queries),
        "assessment_count": len(assessments),
        "answerability": dict(Counter(query.answerability.value for query in queries)),
        "partition": dict(Counter(query.partition for query in queries)),
        "scope": dict(Counter(query.scope.value for query in queries)),
        "temporal_mode": dict(Counter(query.temporal_mode.value for query in queries)),
        "judgments": len(judgments or []),
        "hash": canonical_hash([query.model_dump(mode="json") for query in queries]),
    }


def load_queries(path: Path) -> list[QueryRecord]:
    with path.open("r", encoding="utf-8") as handle:
        return [QueryRecord.model_validate_json(line) for line in handle]


def load_judgments(path: Path) -> list[RelevanceJudgment]:
    with path.open("r", encoding="utf-8") as handle:
        return [RelevanceJudgment.model_validate_json(line) for line in handle]
