from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import RetrievalConfig
from .lexical import BM25SparseIndex, sparse_arrays
from .models import EvidenceRecord, RetrievalHit, SourceRef, TemporalMode

POINT_NAMESPACE = uuid.UUID("de49aff6-fb2b-43d7-8bf0-267dff701ff0")


def point_id(evidence_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, evidence_id))


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    *,
    k: int = 60,
) -> dict[str, dict[str, float | int | None]]:
    fused: dict[str, dict[str, float | int | None]] = defaultdict(
        lambda: {
            "dense_rank": None,
            "dense_score": None,
            "lexical_rank": None,
            "lexical_score": None,
            "fusion_score": 0.0,
        }
    )
    for rank, (evidence_id, score) in enumerate(dense, start=1):
        fused[evidence_id]["dense_rank"] = rank
        fused[evidence_id]["dense_score"] = score
        fused[evidence_id]["fusion_score"] = float(fused[evidence_id]["fusion_score"]) + 1 / (k + rank)
    for rank, (evidence_id, score) in enumerate(lexical, start=1):
        fused[evidence_id]["lexical_rank"] = rank
        fused[evidence_id]["lexical_score"] = score
        fused[evidence_id]["fusion_score"] = float(fused[evidence_id]["fusion_score"]) + 1 / (k + rank)
    return dict(fused)


def classify_temporal_mode(query: str) -> TemporalMode:
    lower = query.lower()
    if "as of" in lower or "before " in lower:
        return TemporalMode.AS_OF
    if any(term in lower for term in ("evolve", "history", "over time", "changed over")):
        return TemporalMode.EVOLUTION
    if any(term in lower for term in ("introduced", "first added", "originally")):
        return TemporalMode.INTRODUCED
    if any(term in lower for term in ("latest", "most recent", "newest")):
        return TemporalMode.LATEST
    if any(term in lower for term in ("current", "now", "currently", "how does")):
        return TemporalMode.CURRENT
    return TemporalMode.NEUTRAL


def temporal_utility(
    timestamp: datetime | None,
    mode: TemporalMode,
    *,
    reference_time: datetime,
    half_life_days: int,
) -> float:
    if timestamp is None or mode == TemporalMode.NEUTRAL:
        return 0.0
    age_days = max(0.0, (reference_time - timestamp).total_seconds() / 86400)
    freshness = math.exp(-math.log(2) * age_days / half_life_days)
    if mode in {TemporalMode.CURRENT, TemporalMode.LATEST, TemporalMode.AS_OF}:
        return freshness
    if mode == TemporalMode.INTRODUCED:
        return 1.0 - freshness
    return 0.0


class HybridRetriever:
    def __init__(
        self,
        records: list[EvidenceRecord],
        dense_vectors: dict[str, np.ndarray],
        lexical: BM25SparseIndex,
        config: RetrievalConfig,
        qdrant_path: Path,
    ):
        from qdrant_client import QdrantClient

        self.records = records
        self.by_id = {record.evidence_id: record for record in records}
        self.dense_vectors = dense_vectors
        self.lexical = lexical
        self.config = config
        self.client = QdrantClient(path=str(qdrant_path))
        self._candidate_cache: dict[
            tuple[str, bytes, str, str | None],
            tuple[list[tuple[str, float]], list[tuple[str, float]]],
        ] = {}

    def build(self, *, recreate: bool = False) -> None:
        from qdrant_client import models

        existing = {item.name for item in self.client.get_collections().collections}
        if self.config.collection in existing and recreate:
            self.client.delete_collection(self.config.collection)
            existing.remove(self.config.collection)
        if self.config.collection not in existing:
            dimension = len(next(iter(self.dense_vectors.values())))
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "lexical": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            for field, schema in (
                ("evidence_id", models.PayloadSchemaType.KEYWORD),
                ("evidence_type", models.PayloadSchemaType.KEYWORD),
                ("session_ids", models.PayloadSchemaType.KEYWORD),
                ("checkpoint_pks", models.PayloadSchemaType.KEYWORD),
                ("commit_shas", models.PayloadSchemaType.KEYWORD),
                ("files", models.PayloadSchemaType.KEYWORD),
                ("branch", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.DATETIME),
                ("partition", models.PayloadSchemaType.KEYWORD),
            ):
                self.client.create_payload_index(self.config.collection, field, schema)

        points = []
        for record in self.records:
            lexical_indices, lexical_values = sparse_arrays(
                self.lexical.doc_vectors.get(record.evidence_id, {})
            )
            payload = {
                "evidence_id": record.evidence_id,
                "evidence_type": record.evidence_type.value,
                "title": record.title,
                "text": record.text,
                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                "session_ids": record.session_ids,
                "checkpoint_pks": record.checkpoint_pks,
                "commit_shas": record.commit_shas,
                "files": record.files,
                "branch": record.branch,
                "partition": record.partition,
                "parent_ids": record.parent_ids,
                "citations": [item.model_dump(mode="json") for item in record.source_refs],
            }
            points.append(
                models.PointStruct(
                    id=point_id(record.evidence_id),
                    vector={
                        "dense": self.dense_vectors[record.evidence_id].tolist(),
                        "lexical": models.SparseVector(
                            indices=lexical_indices, values=lexical_values
                        ),
                    },
                    payload=payload,
                )
            )
            if len(points) >= 512:
                self.client.upsert(self.config.collection, points=points, wait=True)
                points = []
        if points:
            self.client.upsert(self.config.collection, points=points, wait=True)

    def _filter(self, filters: dict[str, Any] | None, as_of: datetime | None):
        from qdrant_client import models

        must = []
        for key, value in (filters or {}).items():
            values = value if isinstance(value, list) else [value]
            must.append(
                models.FieldCondition(
                    key=key, match=models.MatchAny(any=values)
                )
            )
        if as_of is not None:
            must.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.DatetimeRange(lte=as_of),
                )
            )
        return models.Filter(must=must) if must else None

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        temporal_mode: TemporalMode | None = None,
        as_of: datetime | None = None,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        from qdrant_client import models

        mode = temporal_mode or classify_temporal_mode(query)
        query_filter = self._filter(filters, as_of)
        cache_key = (
            query,
            query_vector.tobytes(),
            repr(sorted((filters or {}).items())),
            as_of.isoformat() if as_of else None,
        )
        cached = self._candidate_cache.get(cache_key)
        if cached is None:
            dense_points = self.client.query_points(
                collection_name=self.config.collection,
                query=query_vector.tolist(),
                using="dense",
                query_filter=query_filter,
                limit=self.config.dense_limit,
                with_payload=True,
            ).points
            dense = [
                (str(point.payload["evidence_id"]), float(point.score))
                for point in dense_points
            ]
            lexical_indices, lexical_values = sparse_arrays(
                self.lexical.query_vector(query)
            )
            lexical_points = []
            if lexical_indices:
                lexical_points = self.client.query_points(
                    collection_name=self.config.collection,
                    query=models.SparseVector(
                        indices=lexical_indices,
                        values=lexical_values,
                    ),
                    using="lexical",
                    query_filter=query_filter,
                    limit=self.config.lexical_limit,
                    with_payload=True,
                ).points
            lexical = [
                (str(point.payload["evidence_id"]), float(point.score))
                for point in lexical_points
            ]
            self._candidate_cache[cache_key] = (dense, lexical)
        else:
            dense, lexical = cached
        fused = reciprocal_rank_fusion(dense, lexical)
        timestamps = [record.timestamp for record in self.records if record.timestamp]
        reference = as_of or max(timestamps, default=datetime.now(UTC))

        hits: list[RetrievalHit] = []
        for evidence_id, scores in fused.items():
            record = self.by_id[evidence_id]
            dense_score = scores["dense_score"]
            if dense_score is not None and dense_score < self.config.relevance_floor and scores["lexical_rank"] is None:
                continue
            temporal = temporal_utility(
                record.timestamp,
                mode,
                reference_time=reference,
                half_life_days=self.config.temporal_half_life_days,
            )
            fusion_score = float(scores["fusion_score"])
            final = fusion_score + self.config.temporal_weight * temporal * fusion_score
            group_id = (
                record.topic_key
                or (record.session_ids[0] if record.session_ids else None)
                or evidence_id
            )
            hits.append(
                RetrievalHit(
                    evidence_id=evidence_id,
                    dense_rank=scores["dense_rank"],
                    dense_score=dense_score,
                    lexical_rank=scores["lexical_rank"],
                    lexical_score=scores["lexical_score"],
                    fusion_score=fusion_score,
                    temporal_score=temporal,
                    final_score=final,
                    group_id=group_id,
                    citations=record.source_refs,
                    payload={
                        "title": record.title,
                        "evidence_type": record.evidence_type.value,
                        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                        "temporal_mode": mode.value,
                        "filters": filters or {},
                    },
                )
            )
        hits.sort(key=lambda hit: (-hit.final_score, hit.evidence_id))
        final_limit = limit or self.config.final_limit
        grouped = group_hits(hits, limit=max(final_limit * 3, final_limit))
        if mode == TemporalMode.EVOLUTION:
            grouped = diversify_evolution(grouped, final_limit)
        else:
            grouped = grouped[:final_limit]
        return [
            hit.model_copy(
                update={
                    "payload": {
                        **hit.payload,
                        "final_rank": rank,
                    }
                }
            )
            for rank, hit in enumerate(grouped, start=1)
        ]


def group_hits(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.group_id or hit.evidence_id].append(hit)
    representatives: list[RetrievalHit] = []
    for group_id, members in grouped.items():
        members.sort(key=lambda hit: (-hit.final_score, hit.evidence_id))
        best = members[0]
        citations = []
        seen_citations: set[tuple[Any, ...]] = set()
        for member in members[:3]:
            for citation in member.citations:
                key = (
                    citation.source_file,
                    citation.source_row,
                    citation.event_index,
                    citation.session_id,
                    citation.turn_id,
                    citation.checkpoint_pk,
                    citation.commit_sha,
                )
                if key not in seen_citations:
                    seen_citations.add(key)
                    citations.append(citation)
        representatives.append(
            best.model_copy(
                update={
                    "final_score": best.final_score
                    + sum(member.final_score for member in members[1:3]) * 0.1,
                    "citations": citations,
                    "payload": {
                        **best.payload,
                        "group_size": len(members),
                        "group_member_ids": [member.evidence_id for member in members[:10]],
                    },
                }
            )
        )
    representatives.sort(key=lambda hit: (-hit.final_score, hit.evidence_id))
    return representatives[:limit]


def diversify_evolution(
    hits: list[RetrievalHit],
    limit: int,
    *,
    target_dates: int = 3,
) -> list[RetrievalHit]:
    """Preserve relevance first, then expose useful evidence across distinct dates."""
    if limit <= 0 or not hits:
        return []
    selected = [hits[0]]
    remaining = hits[1:]

    def day(hit: RetrievalHit):
        value = hit.payload.get("timestamp")
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except ValueError:
            return None

    selected_days = {value for value in (day(hit) for hit in selected) if value}
    while remaining and len(selected_days) < min(target_dates, limit):
        dated = [hit for hit in remaining if day(hit) not in selected_days and day(hit)]
        if not dated:
            break
        if selected_days:
            candidate = max(
                dated,
                key=lambda hit: (
                    min(abs((day(hit) - chosen).days) for chosen in selected_days),
                    hit.final_score,
                ),
            )
        else:
            candidate = dated[0]
        selected.append(candidate)
        remaining.remove(candidate)
        selected_days.add(day(candidate))
    selected.extend(hit for hit in hits if hit not in selected)
    return selected[:limit]
