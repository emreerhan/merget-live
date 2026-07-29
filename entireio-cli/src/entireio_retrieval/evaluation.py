from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import (
    AnswerRecord,
    Answerability,
    QueryRecord,
    RelevanceJudgment,
    RetrievalHit,
    TemporalMode,
)


def _qrels(judgments: list[RelevanceJudgment]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgments:
        output[judgment.query_id][judgment.evidence_id] = judgment.grade
    return dict(output)


def dcg(grades: list[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(index + 2)
        for index, grade in enumerate(grades[:k])
    )


def retrieval_metrics(
    rankings: dict[str, list[RetrievalHit | str]],
    judgments: list[RelevanceJudgment],
    queries: list[QueryRecord],
) -> dict[str, Any]:
    qrels = _qrels(judgments)
    ndcgs, reciprocal_ranks, precision5, recall20, recall50 = [], [], [], [], []
    evidence_coverage, claim_coverage = [], []
    temporal_correct, answerability_correct = [], []
    query_by_id = {query.query_id: query for query in queries}

    per_query = {}
    for query_id, query in query_by_id.items():
        result_items = rankings.get(query_id, [])
        ids = [
            item.evidence_id if isinstance(item, RetrievalHit) else item
            for item in result_items
        ]
        relevance = qrels.get(query_id, {})
        grades = [relevance.get(item, 0) for item in ids]
        ideal = sorted(relevance.values(), reverse=True)
        ideal_dcg = dcg(ideal, 10)
        ndcg = dcg(grades, 10) / ideal_dcg if ideal_dcg else (1.0 if not ids else 0.0)
        relevant_ids = {item for item, grade in relevance.items() if grade > 0}
        rr = next((1 / (index + 1) for index, grade in enumerate(grades[:10]) if grade > 0), 0.0)
        p5 = sum(grade > 0 for grade in grades[:5]) / 5
        r20 = len(set(ids[:20]) & relevant_ids) / len(relevant_ids) if relevant_ids else 1.0
        r50 = len(set(ids[:50]) & relevant_ids) / len(relevant_ids) if relevant_ids else 1.0
        covered = set(ids[:20]) & relevant_ids
        expected_claims = {claim.claim_id for claim in query.expected_claims}
        covered_claims = {
            claim_id
            for judgment in judgments
            if judgment.query_id == query_id and judgment.evidence_id in covered
            for claim_id in judgment.supported_claim_ids
        }
        claims = (
            len(covered_claims & expected_claims) / len(expected_claims)
            if expected_claims
            else 1.0
        )
        temporal = _temporal_correctness(query, result_items, relevance)
        answerable_from_results = bool(covered)
        expected_answerable = query.answerability != Answerability.UNSUPPORTED
        answerability = float(answerable_from_results == expected_answerable)
        per_query[query_id] = {
            "ndcg@10": ndcg,
            "mrr@10": rr,
            "precision@5": p5,
            "recall@20": r20,
            "recall@50": r50,
            "evidence_coverage": r20,
            "claim_coverage": claims,
            "temporal_correctness": temporal,
            "answerability_correct": answerability,
        }
        ndcgs.append(ndcg)
        reciprocal_ranks.append(rr)
        precision5.append(p5)
        recall20.append(r20)
        recall50.append(r50)
        evidence_coverage.append(r20)
        claim_coverage.append(claims)
        temporal_correct.append(temporal)
        answerability_correct.append(answerability)
    aggregate = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in {
            "ndcg@10": ndcgs,
            "mrr@10": reciprocal_ranks,
            "precision@5": precision5,
            "recall@20": recall20,
            "recall@50": recall50,
            "evidence_coverage": evidence_coverage,
            "claim_coverage": claim_coverage,
            "temporal_correctness": temporal_correct,
            "answerability_correct": answerability_correct,
        }.items()
    }
    return {"aggregate": aggregate, "per_query": per_query}


def _temporal_correctness(
    query: QueryRecord,
    results: list[RetrievalHit | str],
    relevance: dict[str, int],
) -> float:
    if query.temporal_mode == TemporalMode.NEUTRAL:
        return 1.0
    hits = [item for item in results if isinstance(item, RetrievalHit)]
    relevant_hits = [hit for hit in hits[:10] if relevance.get(hit.evidence_id, 0) > 0]
    if not relevant_hits:
        return 0.0
    if query.temporal_mode in {TemporalMode.CURRENT, TemporalMode.LATEST}:
        return 1.0 if relevant_hits[0].temporal_score >= 0.5 else relevant_hits[0].temporal_score
    if query.temporal_mode == TemporalMode.EVOLUTION:
        dates = {
            hit.payload.get("timestamp", "")[:10]
            for hit in relevant_hits
            if hit.payload.get("timestamp")
        }
        return min(1.0, len(dates) / 3)
    return 1.0


def answer_metrics(
    answers: list[AnswerRecord],
    queries: list[QueryRecord],
    judgments: list[RelevanceJudgment],
) -> dict[str, Any]:
    query_by_id = {query.query_id: query for query in queries}
    relevance = _qrels(judgments)
    per_query = {}
    for answer in answers:
        if not answer.query_id or answer.query_id not in query_by_id:
            continue
        query = query_by_id[answer.query_id]
        expected_texts = [claim.text for claim in query.expected_claims]
        answer_texts = [claim.text for claim in answer.claims]
        claim_coverage = _semantic_claim_coverage(expected_texts, answer_texts)
        cited = {
            citation_id for claim in answer.claims for citation_id in claim.citation_ids
        }
        relevant = {
            evidence_id
            for evidence_id, grade in relevance.get(query.query_id, {}).items()
            if grade > 0
        }
        citation_precision = len(cited & relevant) / len(cited) if cited else (1.0 if not relevant else 0.0)
        citation_recall = len(cited & relevant) / len(relevant) if relevant else 1.0
        answerability_correct = answer.answerability == query.answerability
        false_answer = (
            query.answerability == Answerability.UNSUPPORTED
            and bool(answer.claims)
        )
        per_query[answer.query_id] = {
            "claim_coverage": claim_coverage,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "groundedness": 1.0 if answer.valid else 0.0,
            "answerability_correct": float(answerability_correct),
            "unsupported_false_answer": float(false_answer),
            "temporal_correctness": float(
                answer.temporal_mode == query.temporal_mode
                and not any("temporal" in error.lower() for error in answer.validation_errors)
            ),
        }
    keys = next(iter(per_query.values())).keys() if per_query else []
    aggregate = {
        key: float(np.mean([row[key] for row in per_query.values()]))
        for key in keys
    }
    return {"aggregate": aggregate, "per_query": per_query}


def _semantic_claim_coverage(expected: list[str], actual: list[str], threshold: float = 0.55) -> float:
    if not expected:
        return 1.0
    if not actual:
        return 0.0
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(expected + actual)
    similarity = cosine_similarity(matrix[: len(expected)], matrix[len(expected) :])
    return float(np.mean(np.max(similarity, axis=1) >= threshold))
