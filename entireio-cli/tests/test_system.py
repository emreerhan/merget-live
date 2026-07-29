from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from entireio_retrieval.answering import (
    AnswerGenerator,
    assemble_context,
    deterministic_answer_errors,
)
from entireio_retrieval.benchmark import (
    build_candidate_pool,
    pool_candidates,
    semantic_deduplicate,
    validate_query,
)
from entireio_retrieval.config import (
    EvidenceConfig,
    OpenRouterConfig,
    PilotConfig,
    RetrievalConfig,
    load_config,
)
from entireio_retrieval.cli import (
    _missing_answer_ids,
    _record_manifest,
    _trace_rankings,
)
from entireio_retrieval.evaluation import answer_metrics, retrieval_metrics
from entireio_retrieval.evidence import EvidenceBuilder, bounded_chunks
from entireio_retrieval.io import DatasetReader, build_relationships
from entireio_retrieval.lexical import BM25SparseIndex
from entireio_retrieval.models import (
    AnswerClaim,
    AnswerRecord,
    Answerability,
    AnswerValidationResponse,
    EvidenceRecord,
    EvidenceType,
    ExpectedClaim,
    GeneratedAnswer,
    GeneratedQueryBatch,
    QueryRecord,
    RelevanceJudgment,
    RetrievalHit,
    Scope,
    SourceRef,
    TemporalMode,
)
from entireio_retrieval.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    strict_json_schema,
)
from entireio_retrieval.partition import assign_partitions
from entireio_retrieval.provenance import stable_id
from entireio_retrieval.retrieval import (
    HybridRetriever,
    classify_temporal_mode,
    diversify_evolution,
    reciprocal_rank_fusion,
    temporal_utility,
)
from entireio_retrieval.security import contains_sensitive_text, redact_text
from entireio_retrieval.training import (
    ConflictFreeBatchSampler,
    assert_training_isolation,
    build_training_examples,
)


def evidence(
    identifier: str,
    text: str,
    *,
    partition: str = "train",
    timestamp: datetime | None = None,
    session: str = "session-1",
    checkpoint: str = "checkpoint-1",
    kind: EvidenceType = EvidenceType.CONVERSATION,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type=kind,
        title=identifier,
        text=text,
        timestamp=timestamp,
        source_refs=[
            SourceRef(
                source_file="conversations.parquet",
                source_kind="parquet",
                source_row=1,
                session_id=session,
            )
        ],
        session_ids=[session],
        checkpoint_pks=[checkpoint],
        partition=partition,
        content_hash=f"sha256:{stable_id('content', text)}",
    )


def query(
    identifier: str,
    text: str,
    positive: str,
    *,
    partition: str = "train",
    answerability: Answerability = Answerability.SUPPORTED,
) -> QueryRecord:
    claims = []
    positives = []
    if answerability != Answerability.UNSUPPORTED:
        claims = [
            ExpectedClaim(
                claim_id="claim-1",
                text="The project uses Go.",
                supporting_evidence_ids=[positive],
            )
        ]
        positives = [positive]
    return QueryRecord(
        query_id=identifier,
        query=text,
        category="technology",
        scope=Scope.REPOSITORY,
        temporal_mode=TemporalMode.NEUTRAL,
        answerability=answerability,
        expected_claims=claims,
        primary_positive_ids=positives,
        generation_source_ids=positives,
        partition=partition,
        paraphrase_family=identifier,
    )


def test_reader_reports_missing_and_malformed_sources(tmp_path: Path) -> None:
    schemas = {
        "sessions": {
            "session_id": ["s1"], "repo_id": ["entireio/cli"],
            "created_at": [datetime(2026, 1, 1, tzinfo=UTC)],
            "checkpoint_ids": ['["cp-missing"]'],
        },
        "session_logs": {"session_id": ["s1"], "transcript_path": ["transcripts/s1.jsonl"]},
        "conversations": {
            "turn_id": ["t1"], "session_id": ["s1"], "role": ["user"],
            "turn_type": ["user_prompt"], "content": ["question"], "timestamp": [None],
        },
        "checkpoints": {
            "checkpoint_pk": ["cp1"], "session_pks": ['["s1"]'], "commit_shas": ['["dead"]'],
        },
        "commits": {"checkpoint_pk": ["cp1"], "status": ["error"], "patch": [""]},
    }
    for name, values in schemas.items():
        pq.write_table(pa.table(values), tmp_path / f"{name}.parquet")
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "s1.jsonl").write_text(
        '{"type":"user","timestamp":"2026-01-01T00:00:00Z",'
        '"message":{"role":"user","content":"inspect checkpoint behavior"}}\n'
        "not json\n",
        encoding="utf-8",
    )

    reader = DatasetReader(tmp_path)
    assert reader.validate()["transcripts"] == 1
    assert list(reader.iter_transcript("transcripts/s1.jsonl"))[1][1]["type"] == "malformed"
    relationships = build_relationships(reader)
    assert "checkpoint:cp-missing" in relationships["sessions"]["s1"]["missing"]
    assert relationships["sessions"]["s1"]["__source_row__"] == 0
    records = EvidenceBuilder(reader, EvidenceConfig()).build({"s1"})
    assert any(record.evidence_type == EvidenceType.CHECKPOINT for record in records)
    transcript = next(
        record
        for record in records
        if record.evidence_type == EvidenceType.TRANSCRIPT
    )
    assert transcript.source_refs[0].event_index == 0
    assert transcript.source_refs[0].transcript_path == "transcripts/s1.jsonl"


def test_reader_rejects_transcript_escape(tmp_path: Path) -> None:
    reader = DatasetReader(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        list(reader.iter_transcript("../outside.jsonl"))


def test_bounded_chunks_and_redaction_are_deterministic() -> None:
    text = "a" * 800 + "\n" + "b" * 800
    first = list(bounded_chunks(text, 1000, 100))
    assert first == list(bounded_chunks(text, 1000, 100))
    assert all(len(item) <= 1000 for item in first)
    sensitive = "mail me at dev@example.com token sk-" + "x" * 32 + " /home/alice/private.py"
    redacted = redact_text(sensitive)
    assert "dev@example.com" not in redacted
    assert "/home/alice" not in redacted
    assert not contains_sensitive_text(redacted)


def test_partitioning_preserves_checkpoint_and_topic_isolation() -> None:
    early = datetime(2026, 1, 2, tzinfo=UTC)
    late = datetime(2026, 3, 12, tzinfo=UTC)
    records = [
        evidence("s1", "early session", timestamp=early, session="s1", kind=EvidenceType.SESSION),
        evidence("s1b", "another early session", timestamp=early, session="s1b", checkpoint="cp1b", kind=EvidenceType.SESSION),
        evidence("s2", "late session", timestamp=late, session="s2", checkpoint="cp2", kind=EvidenceType.SESSION),
        evidence("s2b", "another late session", timestamp=late, session="s2b", checkpoint="cp2b", kind=EvidenceType.SESSION),
        EvidenceRecord(
            evidence_id="topic",
            evidence_type=EvidenceType.TOPIC,
            title="topic",
            text="cross lifecycle",
            timestamp=late,
            source_refs=[SourceRef(source_file="sessions.parquet", source_kind="parquet", source_row=0)],
            parent_ids=["s1", "s1b", "s2", "s2b"],
            session_ids=["s1", "s1b", "s2", "s2b"],
            topic_key="src/core",
            content_hash="sha256:topic",
        ),
    ]
    result, report = assign_partitions(
        records,
        PilotConfig(
            train_before="2026-03-01T00:00:00Z",
            validation_before="2026-03-10T00:00:00Z",
        ),
    )
    assert report["sessions"]["train"] == 2
    assert report["sessions"]["evaluation"] == 2
    topics = [item for item in result if item.evidence_type == EvidenceType.TOPIC]
    assert all(len({item.partition}) == 1 for item in topics)
    assert {item.topic_key for item in topics} == {"src/core@train", "src/core@evaluation"}


def test_query_validation_dedup_and_pooling() -> None:
    record = evidence("e1", "The repository primarily uses Go.")
    supported = query("q1", "What language does this repository primarily use?", "e1")
    assert validate_query(supported, {"e1": record}) == []
    unsupported = query(
        "q2",
        "What database does the hosted production service use?",
        "e1",
        answerability=Answerability.UNSUPPORTED,
    )
    assert validate_query(unsupported, {"e1": record}) == []
    duplicate = supported.model_copy(update={"query_id": "q3"})
    assert semantic_deduplicate([supported, duplicate]) == [supported]
    assert pool_candidates({"q1": ["e1", "e2"]}, {"q1": ["e2", "e3"]}) == {
        "q1": ["e1", "e2", "e3"]
    }


def test_candidate_pool_combines_independent_sources() -> None:
    records = [
        evidence("e1", "Go language repository"),
        evidence("e2", "command line architecture", session="s2", checkpoint="cp2"),
    ]

    class Encoder:
        def encode_queries(self, _: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    candidates, traces = build_candidate_pool(
        [query("q1", "What language is used in this repository?", "e1")],
        records,
        Encoder(),
        {
            "e1": np.asarray([1.0, 0.0], dtype=np.float32),
            "e2": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        BM25SparseIndex().fit(records),
    )
    assert candidates["q1"][0] == "e1"
    assert set(traces["q1"]) == {
        "bm25", "pretrained_dense", "metadata_neighbor", "temporally_diverse"
    }


def test_openrouter_structured_cache_never_persists_key(tmp_path: Path) -> None:
    key = "secret-test-key-with-enough-characters"
    key_file = tmp_path / "key.txt"
    key_file.write_text(key, encoding="utf-8")
    key_file.chmod(0o600)
    body = {
        "choices": [{"message": {"content": json.dumps({"queries": []})}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.001},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {key}"
        return httpx.Response(200, json=body)

    client = OpenRouterClient(
        OpenRouterConfig(max_retries=0),
        key_file,
        tmp_path / "cache",
        transport=httpx.MockTransport(handler),
    )
    kwargs = {
        "messages": [{"role": "user", "content": "generate"}],
        "response_model": GeneratedQueryBatch,
        "purpose": "test-generation",
    }
    assert client.complete_structured(**kwargs).queries == []
    assert client.complete_structured(**kwargs).queries == []
    assert client.usage["requests"] == 1
    assert client.usage["cache_hits"] == 1
    assert key not in next((tmp_path / "cache").rglob("*.json")).read_text()


def test_openrouter_retry_exhaustion_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("secret-test-key-with-enough-characters", encoding="utf-8")
    key_file.chmod(0o600)
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500, text="temporary")

    monkeypatch.setattr("entireio_retrieval.openrouter.time.sleep", lambda _: None)
    client = OpenRouterClient(
        OpenRouterConfig(max_retries=2),
        key_file,
        tmp_path / "cache",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenRouterError, match="failed after 3 attempts"):
        client.complete_structured(
            messages=[{"role": "user", "content": "generate"}],
            response_model=GeneratedQueryBatch,
            purpose="retry-test",
        )
    assert requests == 3


def test_provider_schema_marks_every_property_required() -> None:
    schema = strict_json_schema(GeneratedQueryBatch.model_json_schema())
    assert schema["required"] == ["queries"]
    assert "$defs" not in schema
    query_schema = schema["properties"]["queries"]["items"]
    assert set(query_schema["required"]) == set(query_schema["properties"])
    assert query_schema["additionalProperties"] is False
    claim_schema = query_schema["properties"]["expected_claims"]["items"]
    assert claim_schema["properties"]["evidence_quotes"]["properties"] == {}


def test_stage_manifest_records_inputs_code_and_artifacts(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    ).model_copy(
        update={
            "data_dir": tmp_path,
            "derived_dir": tmp_path / "derived",
        }
    )
    input_path = tmp_path / "input.txt"
    artifact_path = tmp_path / "derived" / "output.json"
    input_path.write_text("input", encoding="utf-8")
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"output": true}\n', encoding="utf-8")
    _record_manifest(
        config,
        "test",
        input_paths=[input_path],
        artifact_paths=[artifact_path],
    )
    manifest_path = next((config.derived_dir / "manifests").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_hashes"]
    assert manifest["artifact_hashes"]
    assert manifest["code_version"]


def test_ranking_grouping_filters_and_temporal_modes(tmp_path: Path) -> None:
    now = datetime(2026, 3, 20, tzinfo=UTC)
    records = [
        evidence("go", "Go language CLI implementation", timestamp=now, session="s1"),
        evidence("python", "Python evaluation script", timestamp=now - timedelta(days=60), session="s2", checkpoint="cp2"),
    ]
    vectors = {
        "go": np.asarray([1.0, 0.0], dtype=np.float32),
        "python": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    lexical = BM25SparseIndex().fit(records)
    retriever = HybridRetriever(
        records,
        vectors,
        lexical,
        RetrievalConfig(dense_limit=2, lexical_limit=2, final_limit=2, relevance_floor=-1),
        tmp_path / "qdrant",
    )
    retriever.build()
    hits = retriever.search("Go language", vectors["go"], filters={"session_ids": "s1"})
    assert hits[0].evidence_id == "go"
    assert all(hit.citations for hit in hits)
    fused = reciprocal_rank_fusion([("a", 1.0)], [("b", 1.0), ("a", 0.5)])
    assert fused["a"]["fusion_score"] > fused["b"]["fusion_score"]
    assert classify_temporal_mode("What was introduced first?") == TemporalMode.INTRODUCED
    assert temporal_utility(
        now, TemporalMode.LATEST, reference_time=now, half_life_days=30
    ) == pytest.approx(1.0)
    retriever.client.close()


def test_evolution_diversification_preserves_distinct_dates() -> None:
    hits = [
        RetrievalHit(
            evidence_id=f"e{index}",
            fusion_score=score,
            final_score=score,
            payload={"timestamp": timestamp.isoformat()},
        )
        for index, (score, timestamp) in enumerate(
            (
                (1.0, datetime(2026, 3, 20, tzinfo=UTC)),
                (0.9, datetime(2026, 3, 20, tzinfo=UTC)),
                (0.8, datetime(2026, 3, 1, tzinfo=UTC)),
                (0.7, datetime(2026, 1, 1, tzinfo=UTC)),
            )
        )
    ]
    result = diversify_evolution(hits, 3)
    assert result[0].evidence_id == "e0"
    assert len({hit.payload["timestamp"][:10] for hit in result}) == 3


def test_training_examples_only_admit_graded_zero_negatives() -> None:
    records = [
        evidence("positive", "Go code"),
        evidence("negative", "Unrelated CSS"),
        evidence("eval", "Future evidence", partition="evaluation"),
    ]
    questions = [query("q1", "What language is used in this project?", "positive")]
    judgments = [
        RelevanceJudgment(query_id="q1", evidence_id="positive", grade=3),
        RelevanceJudgment(query_id="q1", evidence_id="negative", grade=0),
    ]
    examples = build_training_examples(questions, records, judgments)
    assert examples[0]["negative_ids"] == ["negative"]
    assert_training_isolation(examples, questions, records)
    leaking = [{**examples[0], "negative_ids": ["eval"]}]
    with pytest.raises(AssertionError, match="non-training evidence"):
        assert_training_isolation(leaking, questions, records)


def test_conflict_free_sampler_separates_related_positives() -> None:
    examples = [
        {
            "positive_id": "a",
            "excluded_positive_ids": ["b"],
        },
        {
            "positive_id": "b",
            "excluded_positive_ids": ["a"],
        },
        {
            "positive_id": "c",
            "excluded_positive_ids": [],
        },
    ]
    batches = list(ConflictFreeBatchSampler(examples, batch_size=3))
    assert not any({0, 1}.issubset(batch) for batch in batches)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]


def test_answer_context_and_citation_validation() -> None:
    record = evidence("e1", "The CLI is implemented in Go.")
    hit = RetrievalHit(
        evidence_id="e1",
        fusion_score=1.0,
        final_score=1.0,
        citations=record.source_refs,
        payload={"group_member_ids": ["e1"]},
    )
    assert assemble_context([hit], {"e1": record}) == [record]
    valid = GeneratedAnswer(
        answerability=Answerability.SUPPORTED,
        claims=[AnswerClaim(text="The CLI uses Go.", citation_ids=["e1"])],
        answer="The captured dataset indicates that the CLI uses Go [e1].",
    )
    assert deterministic_answer_errors(valid, {"e1": record}) == []
    invalid = valid.model_copy(
        update={"claims": [AnswerClaim(text="Uses Rust.", citation_ids=["missing"])]}
    )
    assert deterministic_answer_errors(invalid, {"e1": record})


def test_answer_generator_handles_supported_and_unsupported_paths(tmp_path: Path) -> None:
    record = evidence("e1", "The captured CLI work uses Go.")
    hit = RetrievalHit(
        evidence_id="e1",
        fusion_score=1.0,
        final_score=1.0,
        citations=record.source_refs,
        payload={"group_member_ids": ["e1"]},
    )

    class Client:
        class Config:
            model = "test-model"

        config = Config()

        def complete_structured(self, *, response_model, **_):
            if response_model is GeneratedAnswer:
                return GeneratedAnswer(
                    answerability=Answerability.PARTIAL,
                    claims=[AnswerClaim(text="The captured work uses Go.", citation_ids=["e1"])],
                    answer="The dataset records Go usage [e1].",
                    limitations=["The dataset is not a complete repository snapshot."],
                )
            return AnswerValidationResponse(accepted=True)

    generator = AnswerGenerator(Client(), [record], tmp_path)
    supported = generator.answer("What language is used in the captured work?", [hit])
    assert supported.valid
    assert supported.answerability == Answerability.PARTIAL
    assert supported.citations["e1"].source_file == "conversations.parquet"
    unsupported = generator.answer("What is the production database?", [])
    assert unsupported.answerability == Answerability.UNSUPPORTED
    assert unsupported.valid


def test_retrieval_metrics_cover_standard_measures() -> None:
    question = query("q1", "What language is used in this project?", "e1", partition="evaluation")
    rankings = {
        "q1": [
            RetrievalHit(evidence_id="e1", fusion_score=1.0, final_score=1.0),
            RetrievalHit(evidence_id="e2", fusion_score=0.5, final_score=0.5),
        ]
    }
    judgments = [RelevanceJudgment(query_id="q1", evidence_id="e1", grade=3)]
    metrics = retrieval_metrics(rankings, judgments, [question])
    assert metrics["aggregate"]["mrr@10"] == pytest.approx(1.0)
    assert metrics["aggregate"]["precision@5"] == pytest.approx(0.2)
    assert metrics["aggregate"]["recall@20"] == pytest.approx(1.0)


def test_answer_metrics_cover_grounding_and_unsupported_behavior() -> None:
    supported = query(
        "q1",
        "What language is used in this project?",
        "e1",
        partition="evaluation",
    )
    unsupported = query(
        "q2",
        "What database is used by the hosted production service?",
        "e1",
        partition="evaluation",
        answerability=Answerability.UNSUPPORTED,
    )
    answers = [
        AnswerRecord(
            query_id="q1",
            query=supported.query,
            answerability=Answerability.SUPPORTED,
            temporal_mode=TemporalMode.NEUTRAL,
            claims=[AnswerClaim(text="The project uses Go.", citation_ids=["e1"])],
            answer="The recorded work uses Go [e1].",
            citations={
                "e1": SourceRef(
                    source_file="conversations.parquet",
                    source_kind="parquet",
                    source_row=1,
                )
            },
            valid=True,
        ),
        AnswerRecord(
            query_id="q2",
            query=unsupported.query,
            answerability=Answerability.UNSUPPORTED,
            temporal_mode=TemporalMode.NEUTRAL,
            answer="The indexed dataset does not contain adequate evidence.",
            valid=True,
        ),
    ]
    metrics = answer_metrics(
        answers,
        [supported, unsupported],
        [RelevanceJudgment(query_id="q1", evidence_id="e1", grade=3)],
    )
    assert metrics["aggregate"]["groundedness"] == pytest.approx(1.0)
    assert metrics["aggregate"]["answerability_correct"] == pytest.approx(1.0)
    assert metrics["aggregate"]["unsupported_false_answer"] == pytest.approx(0.0)


def test_trace_artifacts_and_completion_guard(tmp_path: Path) -> None:
    hit = RetrievalHit(
        evidence_id="e1",
        dense_rank=1,
        dense_score=0.9,
        lexical_rank=2,
        lexical_score=1.2,
        fusion_score=0.03,
        temporal_score=0.5,
        final_score=0.04,
        group_id="session-1",
        payload={
            "group_member_ids": ["e1", "e2"],
            "temporal_mode": TemporalMode.CURRENT.value,
            "filters": {"partition": "evaluation"},
        },
    )
    trace = _trace_rankings({"q1": [hit]}, "hybrid")
    assert trace["q1"][0]["dense_score"] == pytest.approx(0.9)
    assert trace["q1"][0]["group_member_ids"] == ["e1", "e2"]
    question = query(
        "q1",
        "What language is used in this project?",
        "e1",
        partition="evaluation",
    )
    assert _missing_answer_ids([question], tmp_path) == ["q1"]
    (tmp_path / "q1.json").write_text(
        AnswerRecord(
            query_id="q1",
            query=question.query,
            answerability=Answerability.UNSUPPORTED,
            temporal_mode=TemporalMode.NEUTRAL,
            answer="No adequate evidence.",
            valid=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    assert _missing_answer_ids([question], tmp_path) == []
