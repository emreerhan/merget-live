from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PilotConfig(ConfigModel):
    size: int = Field(default=275, ge=1, le=870)
    seed: int = 20260729
    train_before: str = "2026-03-01T00:00:00Z"
    validation_before: str = "2026-03-10T00:00:00Z"


class EvidenceConfig(ConfigModel):
    max_chars: int = Field(default=8000, ge=500)
    chunk_chars: int = Field(default=5000, ge=500)
    chunk_overlap: int = Field(default=300, ge=0)
    include_thinking: bool = True
    include_tool_results: bool = True

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_is_bounded(cls, value: int, info):
        chunk = info.data.get("chunk_chars", 5000)
        if value >= chunk:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        return value


class OpenRouterConfig(ConfigModel):
    model: str = "openai/gpt-5.6-luna"
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    base_url: str = "https://openrouter.ai/api/v1"
    max_concurrency: int = Field(default=4, ge=1, le=8)
    max_retries: int = Field(default=4, ge=0, le=10)
    timeout_seconds: int = Field(default=120, ge=10)
    exclude_reasoning: bool = True


class EmbeddingConfig(ConfigModel):
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    dimension: int = 384
    max_sequence_length: int = 256
    batch_size: int = 64
    query_instruction: str
    index_conversation_types: list[str] = Field(
        default_factory=lambda: ["user_prompt", "assistant_response", "summary"]
    )


class TrainingConfig(ConfigModel):
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    freeze_lower_layers: int = Field(default=4, ge=0)
    early_stopping_patience: int = Field(default=1, ge=0)


class RetrievalConfig(ConfigModel):
    collection: str = "entireio_cli_evidence"
    dense_limit: int = 50
    lexical_limit: int = 50
    final_limit: int = 10
    relevance_floor: float = Field(default=0.15, ge=-1, le=1)
    temporal_weight: float = Field(default=0.15, ge=0, le=0.5)
    temporal_half_life_days: int = Field(default=30, ge=1)


class AppConfig(ConfigModel):
    data_dir: Path
    derived_dir: Path
    key_file: Path
    pilot: PilotConfig
    evidence: EvidenceConfig
    openrouter: OpenRouterConfig
    embedding: EmbeddingConfig
    training: TrainingConfig
    retrieval: RetrievalConfig

    def resolve(self, config_path: Path) -> "AppConfig":
        base = config_path.parent.parent.resolve()
        updates = {}
        for field in ("data_dir", "derived_dir", "key_file"):
            value = getattr(self, field)
            updates[field] = value if value.is_absolute() else (base / value).resolve()
        return self.model_copy(update=updates)


def load_config(path: Path | str) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AppConfig.model_validate(data).resolve(config_path)
