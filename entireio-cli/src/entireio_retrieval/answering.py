from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .benchmark import evidence_packet
from .models import (
    AnswerRecord,
    AnswerValidationResponse,
    Answerability,
    EvidenceRecord,
    GeneratedAnswer,
    RetrievalHit,
    TemporalMode,
)
from .openrouter import OpenRouterClient
from .prompts import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_VALIDATOR_PROMPT_VERSION,
    ANSWER_VALIDATOR_SYSTEM_PROMPT,
)
from .provenance import stable_id, write_json_atomic


def assemble_context(
    hits: list[RetrievalHit],
    evidence_by_id: dict[str, EvidenceRecord],
    *,
    max_chars: int = 40000,
) -> list[EvidenceRecord]:
    selected: list[EvidenceRecord] = []
    used = 0
    seen: set[str] = set()
    for hit in hits:
        member_ids = hit.payload.get("group_member_ids", [hit.evidence_id])
        for evidence_id in member_ids:
            if evidence_id in seen or evidence_id not in evidence_by_id:
                continue
            record = evidence_by_id[evidence_id]
            if selected and used + len(record.text) > max_chars:
                return selected
            selected.append(record)
            seen.add(evidence_id)
            used += len(record.text)
    return selected


def deterministic_answer_errors(
    answer: GeneratedAnswer,
    evidence_by_id: dict[str, EvidenceRecord],
) -> list[str]:
    errors = []
    cited = [citation_id for claim in answer.claims for citation_id in claim.citation_ids]
    missing = sorted(set(cited) - evidence_by_id.keys())
    if missing:
        errors.append(f"missing citations: {missing}")
    if answer.answerability == Answerability.UNSUPPORTED and answer.claims:
        errors.append("unsupported answer contains factual claims")
    if answer.answerability != Answerability.UNSUPPORTED:
        if not answer.claims:
            errors.append("supported/partial answer contains no claims")
        for claim in answer.claims:
            if not claim.citation_ids:
                errors.append(f"claim lacks citation: {claim.text}")
    return errors


class AnswerGenerator:
    def __init__(
        self,
        client: OpenRouterClient,
        evidence: list[EvidenceRecord],
        artifact_dir: Path,
    ):
        self.client = client
        self.by_id = {record.evidence_id: record for record in evidence}
        self.artifact_dir = artifact_dir

    def answer(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        query_id: str | None = None,
        temporal_mode: TemporalMode = TemporalMode.NEUTRAL,
        max_regenerations: int = 1,
    ) -> AnswerRecord:
        context = assemble_context(hits, self.by_id)
        if not context:
            result = AnswerRecord(
                query_id=query_id,
                query=query,
                answerability=Answerability.UNSUPPORTED,
                temporal_mode=temporal_mode,
                answer="The indexed dataset does not contain adequate evidence to answer this question.",
                limitations=["No retrievable evidence passed the support threshold."],
                valid=True,
                provenance={"mode": "deterministic_abstention"},
            )
            self._persist(result)
            return result

        payload = {
            "prompt_version": ANSWER_PROMPT_VERSION,
            "query": query,
            "temporal_mode": temporal_mode.value,
            "requirements": {
                "dataset_scoped_language": True,
                "cite_every_factual_claim": True,
                "do_not_use_background_knowledge": True,
            },
            **evidence_packet(context, max_chars=40000),
        }
        last_errors: list[str] = []
        for attempt in range(max_regenerations + 1):
            generated = self.client.complete_structured(
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {**payload, "prior_validation_errors": last_errors},
                            default=str,
                        ),
                    },
                ],
                response_model=GeneratedAnswer,
                purpose=f"answer-{stable_id('answer', query, attempt)}",
                force=attempt > 0,
            )
            errors = deterministic_answer_errors(generated, self.by_id)
            validation = self._validate(query, temporal_mode, context, generated, attempt)
            errors.extend(validation.unsupported_claims)
            errors.extend(
                f"invalid citation: {item}" for item in validation.invalid_citation_ids
            )
            if validation.answerability_issue:
                errors.append(validation.answerability_issue)
            if validation.temporal_issue:
                errors.append(validation.temporal_issue)
            if not validation.accepted and not errors:
                errors.append("groundedness validator rejected the answer")
            if not errors:
                citations = {
                    citation_id: self.by_id[citation_id].source_refs[0]
                    for claim in generated.claims
                    for citation_id in claim.citation_ids
                }
                result = AnswerRecord(
                    query_id=query_id,
                    query=query,
                    temporal_mode=temporal_mode,
                    **generated.model_dump(),
                    citations=citations,
                    valid=True,
                    provenance={
                        "generator_prompt": ANSWER_PROMPT_VERSION,
                        "validator_prompt": ANSWER_VALIDATOR_PROMPT_VERSION,
                        "model": self.client.config.model,
                        "attempt": attempt,
                        "retrieved_evidence_ids": [record.evidence_id for record in context],
                    },
                )
                self._persist(result)
                return result
            last_errors = errors
        result = AnswerRecord(
            query_id=query_id,
            query=query,
            answerability=Answerability.UNSUPPORTED,
            temporal_mode=temporal_mode,
            answer="The retrieved evidence could not support a citation-valid answer.",
            limitations=last_errors,
            valid=False,
            validation_errors=last_errors,
            provenance={"attempts": max_regenerations + 1},
        )
        self._persist(result)
        return result

    def _validate(
        self,
        query: str,
        temporal_mode: TemporalMode,
        context: list[EvidenceRecord],
        generated: GeneratedAnswer,
        attempt: int,
    ) -> AnswerValidationResponse:
        payload = {
            "prompt_version": ANSWER_VALIDATOR_PROMPT_VERSION,
            "query": query,
            "temporal_mode": temporal_mode.value,
            "answer": generated.model_dump(mode="json"),
            **evidence_packet(context, max_chars=40000),
        }
        return self.client.complete_structured(
            messages=[
                {"role": "system", "content": ANSWER_VALIDATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_model=AnswerValidationResponse,
            purpose=f"answer-validator-{stable_id('validation', query, attempt)}",
        )

    def _persist(self, result: AnswerRecord) -> None:
        identifier = result.query_id or stable_id("query", result.query)
        write_json_atomic(self.artifact_dir / f"{identifier}.json", result)

