from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = {
    "sessions": {"session_id", "repo_id", "created_at", "checkpoint_ids"},
    "session_logs": {"session_id", "transcript_path"},
    "conversations": {"turn_id", "session_id", "role", "turn_type", "content"},
    "checkpoints": {"checkpoint_pk", "session_pks", "commit_shas"},
    "commits": {"checkpoint_pk", "status", "patch"},
}


class DatasetReader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()

    def validate(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, required in REQUIRED_COLUMNS.items():
            path = self.data_dir / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            missing = required - columns
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            counts[name] = parquet.metadata.num_rows
        transcript_dir = self.data_dir / "transcripts"
        if not transcript_dir.is_dir():
            raise FileNotFoundError(transcript_dir)
        counts["transcripts"] = sum(1 for _ in transcript_dir.glob("*.jsonl"))
        return counts

    def iter_table(
        self, name: str, *, columns: list[str] | None = None, batch_size: int = 2048
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        if name not in REQUIRED_COLUMNS:
            raise KeyError(f"unsupported dataset table: {name}")
        path = self.data_dir / f"{name}.parquet"
        row_number = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
            for row in batch.to_pylist():
                yield row_number, row
                row_number += 1

    def read_table(self, name: str, *, columns: list[str] | None = None) -> pa.Table:
        if name not in REQUIRED_COLUMNS:
            raise KeyError(f"unsupported dataset table: {name}")
        return pq.read_table(self.data_dir / f"{name}.parquet", columns=columns)

    def iter_transcript(self, relative_path: str) -> Iterator[tuple[int, dict[str, Any]]]:
        path = (self.data_dir / relative_path).resolve()
        if self.data_dir not in path.parents:
            raise ValueError(f"transcript escapes data directory: {relative_path}")
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for event_index, line in enumerate(handle):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    yield event_index, {"type": "malformed", "raw": line.rstrip("\n")}
                    continue
                yield event_index, value


def parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def build_relationships(reader: DatasetReader) -> dict[str, Any]:
    sessions: dict[str, dict[str, Any]] = {}
    for row_number, row in reader.iter_table("sessions"):
        sessions[row["session_id"]] = {
            **row,
            "__source_row__": row_number,
            "checkpoint_ids_parsed": parse_json_list(row.get("checkpoint_ids")),
            "conversation_turn_ids": [],
            "commit_shas_resolved": [],
            "missing": [],
        }

    logs: dict[str, dict[str, Any]] = {
        row["session_id"]: {**row, "__source_row__": row_number}
        for row_number, row in reader.iter_table("session_logs")
    }
    checkpoints: dict[str, dict[str, Any]] = {}
    for row_number, row in reader.iter_table("checkpoints"):
        checkpoints[row["checkpoint_pk"]] = {
            **row,
            "__source_row__": row_number,
            "session_ids_parsed": parse_json_list(row.get("session_pks")),
            "commit_shas_parsed": parse_json_list(row.get("commit_shas")),
        }

    commits_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row_number, row in reader.iter_table("commits"):
        item = {**row, "__source_row__": row_number}
        commits_by_checkpoint.setdefault(str(row.get("checkpoint_pk")), []).append(item)

    for _, row in reader.iter_table(
        "conversations", columns=["turn_id", "session_id", "timestamp"]
    ):
        session = sessions.get(row["session_id"])
        if session is not None:
            session["conversation_turn_ids"].append(row["turn_id"])

    for session_id, session in sessions.items():
        log = logs.get(session_id)
        if log is None:
            session["missing"].append("session_log")
        else:
            session["session_log"] = log
            transcript_path = log.get("transcript_path")
            if not transcript_path or not (reader.data_dir / transcript_path).is_file():
                session["missing"].append("transcript")
        for checkpoint_pk in session["checkpoint_ids_parsed"]:
            checkpoint = checkpoints.get(checkpoint_pk)
            if checkpoint is None:
                session["missing"].append(f"checkpoint:{checkpoint_pk}")
                continue
            for commit in commits_by_checkpoint.get(checkpoint_pk, []):
                sha = commit.get("commit_sha")
                if sha:
                    session["commit_shas_resolved"].append(sha)
        if not session["conversation_turn_ids"]:
            session["missing"].append("conversations")

    return {
        "sessions": sessions,
        "session_logs": logs,
        "checkpoints": checkpoints,
        "commits_by_checkpoint": commits_by_checkpoint,
    }
