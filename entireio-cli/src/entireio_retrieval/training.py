from __future__ import annotations

import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from .config import EmbeddingConfig, TrainingConfig
from .models import EvidenceRecord, QueryRecord, RelevanceJudgment
from .provenance import canonical_hash, write_json_atomic


def build_training_examples(
    queries: list[QueryRecord],
    evidence: list[EvidenceRecord],
    judgments: list[RelevanceJudgment],
    *,
    partitions: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_partitions = partitions or {"train"}
    by_id = {record.evidence_id: record for record in evidence}
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgments:
        grades[judgment.query_id][judgment.evidence_id] = judgment.grade
    output = []
    for query in queries:
        if query.partition not in allowed_partitions:
            continue
        positives = list(
            dict.fromkeys(query.primary_positive_ids + query.supporting_positive_ids)
        )
        positives = [
            evidence_id for evidence_id in positives
            if evidence_id in by_id and grades.get(query.query_id, {}).get(evidence_id, 3) > 0
        ]
        known_related = {
            evidence_id for evidence_id, grade in grades.get(query.query_id, {}).items()
            if grade > 0
        } | set(positives)
        negatives = [
            evidence_id for evidence_id, grade in grades.get(query.query_id, {}).items()
            if grade == 0 and evidence_id not in known_related and evidence_id in by_id
        ]
        for positive in positives:
            output.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "positive_id": positive,
                    "positive": by_id[positive].text,
                    "negative_ids": negatives[:3],
                    "negatives": [by_id[item].text for item in negatives[:3]],
                    "excluded_positive_ids": sorted(known_related - {positive}),
                    "query_provenance": query.provenance,
                }
            )
    return output


def assert_training_isolation(
    examples: list[dict[str, Any]],
    queries: list[QueryRecord],
    evidence: list[EvidenceRecord],
) -> None:
    query_partition = {query.query_id: query.partition for query in queries}
    evidence_partition = {record.evidence_id: record.partition for record in evidence}
    errors = []
    for example in examples:
        if query_partition.get(example["query_id"]) != "train":
            errors.append(f"non-training query {example['query_id']}")
        for evidence_id in [example["positive_id"], *example["negative_ids"]]:
            if evidence_partition.get(evidence_id) != "train":
                errors.append(f"non-training evidence {evidence_id}")
    if errors:
        raise AssertionError("training leakage: " + ", ".join(errors[:20]))


def freeze_lower_layers(model, count: int) -> int:
    transformer = model._first_module().auto_model
    layers = None
    for path in (
        ("encoder", "layer"),
        ("transformer", "layer"),
    ):
        current = transformer
        try:
            for name in path:
                current = getattr(current, name)
            layers = current
            break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError("unable to locate transformer layers for freezing")
    frozen = 0
    for layer in list(layers)[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
        frozen += 1
    return frozen


def train_encoder(
    examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    embedding_config: EmbeddingConfig,
    training_config: TrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    from sentence_transformers import InputExample, SentenceTransformer, losses

    if not examples:
        raise ValueError("no training examples")
    state_path = output_dir / "training-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"completed_epochs": 0, "best_epoch": 0, "best_score": None}
    )
    resume_model = (
        output_dir / "checkpoints" / f"epoch-{state['completed_epochs']}"
    )
    model = SentenceTransformer(
        str(resume_model if resume_model.is_dir() else embedding_config.model),
        revision=None if resume_model.is_dir() else embedding_config.revision,
        device="cpu",
        local_files_only=True,
    )
    model.max_seq_length = embedding_config.max_sequence_length
    frozen = freeze_lower_layers(model, training_config.freeze_lower_layers)
    trainable_examples = [example for example in examples if example["negatives"]]
    if not trainable_examples:
        raise ValueError("no training examples with grounded hard negatives")
    # The legacy SentenceTransformers adapter transposes all InputExample.texts
    # with zip(), which silently truncates every sample to the shortest sample.
    # Give every row an equal number of explicit negatives so a single row with
    # no/too few negatives cannot drop hard-negative columns for the full run.
    hard_negative_count = max(len(example["negatives"]) for example in trainable_examples)
    samples = []
    for example in trainable_examples:
        negatives = list(example["negatives"])
        negatives.extend([negatives[-1]] * (hard_negative_count - len(negatives)))
        texts = [
            embedding_config.query_instruction + example["query"],
            example["positive"],
            *negatives,
        ]
        samples.append(InputExample(texts=texts))
    batch_sampler = ConflictFreeBatchSampler(trainable_examples, training_config.batch_size)
    loader = DataLoader(samples, batch_sampler=batch_sampler)
    # SentenceTransformers.fit adapts legacy DataLoaders into a Trainer dataset
    # and reads this attribute only to configure the Trainer batch size. PyTorch
    # sets it to None whenever batch_sampler is supplied, even though our sampler
    # has an explicit maximum batch size.
    loader.__dict__["batch_size"] = training_config.batch_size
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    patience = 0
    completed_epochs = int(state["completed_epochs"])
    best_score = state["best_score"]
    best_epoch = int(state["best_epoch"])
    for epoch in range(completed_epochs + 1, training_config.epochs + 1):
        loss = losses.MultipleNegativesRankingLoss(model)
        checkpoint = output_dir / "checkpoints" / f"epoch-{epoch}"
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            optimizer_params={"lr": training_config.learning_rate},
            warmup_steps=max(1, len(loader) // 10),
            output_path=str(checkpoint),
            show_progress_bar=True,
        )
        score = _validation_margin(
            model,
            validation_examples or examples[: min(100, len(examples))],
            query_instruction=embedding_config.query_instruction,
        )
        improved = best_score is None or score > float(best_score)
        if improved:
            best_score, best_epoch, patience = score, epoch, 0
            model.save(str(output_dir))
        else:
            patience += 1
        state = {
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "validation_margin": score,
        }
        write_json_atomic(state_path, state)
        if patience > training_config.early_stopping_patience:
            break
    report = {
        "examples": len(examples),
        "trainable_examples": len(trainable_examples),
        "hard_negatives_per_example": hard_negative_count,
        "epochs": training_config.epochs,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "frozen_layers": frozen,
        "conflict_free_batches": len(batch_sampler),
        "best_epoch": best_epoch,
        "best_validation_margin": best_score,
        "completed_epochs": state["completed_epochs"],
        "elapsed_seconds": time.monotonic() - started,
        "model": embedding_config.model,
        "revision": embedding_config.revision,
        "resolved_revision": getattr(
            model._first_module().auto_model.config, "_commit_hash", None
        )
        or embedding_config.revision,
        "data_hash": canonical_hash(examples),
        "embedding_config_hash": canonical_hash(
            embedding_config.model_dump(mode="json")
        ),
        "training_config_hash": canonical_hash(
            training_config.model_dump(mode="json")
        ),
        "query_prompt_versions": sorted(
            {
                str(example["query_provenance"].get("generator_prompt"))
                for example in examples
                if example["query_provenance"].get("generator_prompt")
            }
        ),
        "cpu_count": os.cpu_count(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json_atomic(output_dir / "training-report.json", report)
    return report


class ConflictFreeBatchSampler:
    """Prevent a known positive from becoming another query's in-batch negative."""

    def __init__(self, examples: list[dict[str, Any]], batch_size: int):
        self.examples = examples
        self.batch_size = batch_size
        self.batches = self._build()

    def _build(self) -> list[list[int]]:
        remaining = list(range(len(self.examples)))
        batches: list[list[int]] = []
        while remaining:
            batch: list[int] = []
            deferred: list[int] = []
            for index in remaining:
                candidate = self.examples[index]
                candidate_excluded = set(candidate["excluded_positive_ids"])
                candidate_positive = candidate["positive_id"]
                conflict = any(
                    candidate_positive
                    in set(self.examples[chosen]["excluded_positive_ids"])
                    or self.examples[chosen]["positive_id"] in candidate_excluded
                    for chosen in batch
                )
                if not conflict and len(batch) < self.batch_size:
                    batch.append(index)
                else:
                    deferred.append(index)
            batches.append(batch)
            remaining = deferred
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _validation_margin(
    model,
    examples: list[dict[str, Any]],
    *,
    query_instruction: str = "",
) -> float:
    if not examples:
        return 0.0
    queries = [query_instruction + example["query"] for example in examples]
    positives = [example["positive"] for example in examples]
    query_vectors = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    positive_vectors = model.encode(positives, normalize_embeddings=True, convert_to_numpy=True)
    positive_scores = np.sum(query_vectors * positive_vectors, axis=1)
    margins = []
    for index, example in enumerate(examples):
        if example["negatives"]:
            negative = model.encode(
                [example["negatives"][0]],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
            margins.append(float(positive_scores[index] - query_vectors[index] @ negative))
        else:
            margins.append(float(positive_scores[index]))
    return float(np.mean(margins))
