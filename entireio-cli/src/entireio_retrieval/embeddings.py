from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EmbeddingConfig
from .models import EvidenceRecord
from .provenance import canonical_hash, write_json_atomic


class DenseEncoder:
    def __init__(self, config: EmbeddingConfig, model_path: str | Path | None = None):
        from sentence_transformers import SentenceTransformer

        name = str(model_path or config.model)
        self.config = config
        self.model_identity = name
        self.model = SentenceTransformer(
            name,
            revision=None if model_path else config.revision,
            device="cpu",
            local_files_only=True,
        )
        self.model.max_seq_length = config.max_sequence_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension != config.dimension:
            raise ValueError(
                f"encoder dimension {self.dimension} does not match configured {config.dimension}"
            )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefixed = [self.config.query_instruction + query for query in queries]
        return self.encode_documents(prefixed)

    def encode_records(
        self,
        records: list[EvidenceRecord],
        cache_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        fingerprint_payload = {
            "model": self.config.model,
            "model_identity": self.model_identity,
            "revision": self.config.revision,
            "max_sequence_length": self.config.max_sequence_length,
            "records": [(record.evidence_id, record.content_hash) for record in records],
        }
        fingerprint = canonical_hash(fingerprint_payload)
        legacy_fingerprint = canonical_hash(
            {
                key: value
                for key, value in fingerprint_payload.items()
                if key != "model_identity"
            }
        )
        if cache_path and cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            metadata = json.loads(str(cached["metadata"]))
            cached_fingerprint = metadata.get("fingerprint")
            base_model_cache = self.model_identity == self.config.model
            same_model_cache = metadata.get("model_identity") == self.model_identity
            if cached_fingerprint == fingerprint or (
                base_model_cache and cached_fingerprint == legacy_fingerprint
            ):
                ids = [str(item) for item in cached["ids"]]
                return dict(zip(ids, cached["vectors"]))
            if base_model_cache or same_model_cache:
                cached_ids = [str(item) for item in cached["ids"]]
                cached_vectors = {
                    evidence_id: vector
                    for evidence_id, vector in zip(cached_ids, cached["vectors"])
                }
                requested = {record.evidence_id for record in records}
                reusable = {
                    evidence_id: vector
                    for evidence_id, vector in cached_vectors.items()
                    if evidence_id in requested
                }
                missing = [
                    record for record in records if record.evidence_id not in reusable
                ]
                if missing:
                    additions = self.encode_documents(
                        [record.text for record in missing]
                    )
                    reusable.update(
                        {
                            record.evidence_id: vector
                            for record, vector in zip(missing, additions)
                        }
                    )
                result = {
                    record.evidence_id: reusable[record.evidence_id]
                    for record in records
                }
                self._save_cache(cache_path, records, result, fingerprint)
                return result
        vectors = self.encode_documents([record.text for record in records])
        result = {record.evidence_id: vector for record, vector in zip(records, vectors)}
        if cache_path:
            self._save_cache(cache_path, records, result, fingerprint)
        return result

    def _save_cache(
        self,
        cache_path: Path,
        records: list[EvidenceRecord],
        vectors: dict[str, np.ndarray],
        fingerprint: str,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            ids=np.asarray([record.evidence_id for record in records]),
            vectors=np.vstack([vectors[record.evidence_id] for record in records]),
            metadata=json.dumps(
                {
                    "fingerprint": fingerprint,
                    "dimension": self.dimension,
                    "model_identity": self.model_identity,
                }
            ),
        )


def select_index_records(
    records: list[EvidenceRecord],
    config: EmbeddingConfig,
    required_ids: set[str] | None = None,
) -> list[EvidenceRecord]:
    """Keep the initial CPU index compact while retaining the full evidence store."""
    selected = [
        record
        for record in records
        if (
            record.evidence_type.value not in {"conversation", "transcript"}
            or (
                record.evidence_type.value == "conversation"
                and record.metadata.get("turn_type")
                in config.index_conversation_types
            )
            or record.evidence_id in (required_ids or set())
        )
    ]
    if not selected:
        raise ValueError("embedding index selection produced no records")
    return selected


def cosine_rank(
    query_vector: np.ndarray,
    vectors: dict[str, np.ndarray],
    limit: int = 50,
) -> list[tuple[str, float]]:
    if not vectors:
        return []
    ids = list(vectors)
    matrix = np.vstack([vectors[item] for item in ids])
    scores = matrix @ query_vector
    indices = np.argsort(-scores)[:limit]
    return [(ids[index], float(scores[index])) for index in indices]
