from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceType(StrEnum):
    CONVERSATION = "conversation"
    TRANSCRIPT = "transcript"
    CHECKPOINT = "checkpoint"
    COMMIT = "commit"
    SESSION = "session"
    TOPIC = "topic"


class Answerability(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partially_supported"
    UNSUPPORTED = "unsupported_by_dataset"


class TemporalMode(StrEnum):
    CURRENT = "current_state"
    LATEST = "latest_change"
    EVOLUTION = "historical_evolution"
    INTRODUCED = "introduced_when"
    AS_OF = "as_of_date"
    NEUTRAL = "time_neutral"


class Scope(StrEnum):
    REPOSITORY = "repository"
    SUBSYSTEM = "subsystem"
    CHANGE = "change"


class SourceRef(StrictModel):
    source_file: str
    source_kind: Literal["parquet", "transcript", "derived"]
    source_row: int | None = None
    event_index: int | None = None
    session_id: str | None = None
    turn_id: str | None = None
    checkpoint_pk: str | None = None
    commit_sha: str | None = None
    transcript_path: str | None = None


class EvidenceRecord(StrictModel):
    evidence_id: str
    evidence_type: EvidenceType
    title: str
    text: str
    timestamp: datetime | None = None
    source_refs: list[SourceRef]
    parent_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    checkpoint_pks: list[str] = Field(default_factory=list)
    commit_shas: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    branch: str | None = None
    agent: str | None = None
    topic_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    partition: Literal["train", "validation", "evaluation"] | None = None
    content_hash: str

    @field_validator("source_refs")
    @classmethod
    def require_source(cls, value: list[SourceRef]) -> list[SourceRef]:
        if not value:
            raise ValueError("evidence must retain at least one source reference")
        return value


class ExpectedClaim(StrictModel):
    claim_id: str
    text: str
    supporting_evidence_ids: list[str]
    evidence_quotes: dict[str, str] = Field(default_factory=dict)


class QueryRecord(StrictModel):
    query_id: str
    query: str
    category: str
    scope: Scope
    temporal_mode: TemporalMode
    as_of: datetime | None = None
    answerability: Answerability
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    primary_positive_ids: list[str] = Field(default_factory=list)
    supporting_positive_ids: list[str] = Field(default_factory=list)
    generation_source_ids: list[str] = Field(default_factory=list)
    partition: Literal["train", "validation", "evaluation"]
    paraphrase_family: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def standalone_query(cls, value: str) -> str:
        words = value.split()
        if not 4 <= len(words) <= 40:
            raise ValueError("query must contain 4 to 40 words")
        return value.strip()


class CriticScore(StrictModel):
    groundedness: int = Field(ge=1, le=5)
    answerability: int = Field(ge=1, le=5)
    evidence_sufficiency: int = Field(ge=1, le=5)
    naturalness: int = Field(ge=1, le=5)
    ambiguity_issue: str | None = None
    leakage_issue: str | None = None
    temporal_issue: str | None = None
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class RelevanceJudgment(StrictModel):
    query_id: str
    evidence_id: str
    grade: int = Field(ge=0, le=3)
    supported_claim_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(StrictModel):
    evidence_id: str
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_score: float
    temporal_score: float = 0.0
    final_score: float
    group_id: str | None = None
    citations: list[SourceRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class AnswerClaim(StrictModel):
    text: str
    citation_ids: list[str]


class AnswerRecord(StrictModel):
    query_id: str | None = None
    query: str
    answerability: Answerability
    temporal_mode: TemporalMode
    claims: list[AnswerClaim] = Field(default_factory=list)
    answer: str
    limitations: list[str] = Field(default_factory=list)
    citations: dict[str, SourceRef] = Field(default_factory=dict)
    valid: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: str
    stage: str
    created_at: datetime
    input_hashes: dict[str, str]
    config_hash: str
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    code_version: str | None = None
    model_settings: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int | float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class GeneratedQueryBatch(StrictModel):
    queries: list[QueryRecord]


class CriticResponse(StrictModel):
    query_id: str
    score: CriticScore


class JudgmentBatch(StrictModel):
    judgments: list[RelevanceJudgment]


class GeneratedAnswer(StrictModel):
    answerability: Answerability
    claims: list[AnswerClaim]
    answer: str
    limitations: list[str] = Field(default_factory=list)


class AnswerValidationResponse(StrictModel):
    accepted: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    answerability_issue: str | None = None
    temporal_issue: str | None = None
