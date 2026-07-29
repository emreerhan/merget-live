from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .models import EvidenceRecord
from .provenance import write_json_atomic

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25SparseIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.doc_vectors: dict[str, dict[int, float]] = {}

    def fit(self, records: list[EvidenceRecord]) -> "BM25SparseIndex":
        documents = {record.evidence_id: tokenize(record.text) for record in records}
        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))
        count = max(1, len(documents))
        self.vocabulary = {
            token: index for index, token in enumerate(sorted(document_frequency))
        }
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.avgdl = sum(map(len, documents.values())) / count
        self.doc_vectors = {
            evidence_id: self._document_vector(tokens)
            for evidence_id, tokens in documents.items()
        }
        return self

    def _document_vector(self, tokens: list[str]) -> dict[int, float]:
        frequencies = Counter(tokens)
        length = max(1, len(tokens))
        output = {}
        for token, frequency in frequencies.items():
            if token not in self.vocabulary:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / max(self.avgdl, 1)
            )
            weight = self.idf[token] * frequency * (self.k1 + 1) / denominator
            output[self.vocabulary[token]] = float(weight)
        return output

    def query_vector(self, query: str) -> dict[int, float]:
        frequencies = Counter(tokenize(query))
        return {
            self.vocabulary[token]: float(self.idf[token] * (1 + math.log(frequency)))
            for token, frequency in frequencies.items()
            if token in self.vocabulary
        }

    def rank(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        query_vector = self.query_vector(query)
        scores = []
        for evidence_id, document in self.doc_vectors.items():
            score = sum(query_vector.get(index, 0.0) * value for index, value in document.items())
            if score > 0:
                scores.append((evidence_id, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:limit]

    def save(self, path: Path) -> None:
        write_json_atomic(
            path,
            {
                "k1": self.k1,
                "b": self.b,
                "avgdl": self.avgdl,
                "vocabulary": self.vocabulary,
                "idf": self.idf,
                "doc_vectors": {
                    evidence_id: {str(index): value for index, value in vector.items()}
                    for evidence_id, vector in self.doc_vectors.items()
                },
            },
        )

    @classmethod
    def load(cls, path: Path) -> "BM25SparseIndex":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=data["k1"], b=data["b"])
        index.avgdl = data["avgdl"]
        index.vocabulary = {key: int(value) for key, value in data["vocabulary"].items()}
        index.idf = {key: float(value) for key, value in data["idf"].items()}
        index.doc_vectors = {
            evidence_id: {int(key): float(value) for key, value in vector.items()}
            for evidence_id, vector in data["doc_vectors"].items()
        }
        return index


def sparse_arrays(vector: dict[int, float]) -> tuple[list[int], list[float]]:
    ordered = sorted(vector.items())
    return [item[0] for item in ordered], [item[1] for item in ordered]

