from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .models import RunManifest


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return canonical_hash(
            {
                str(item.relative_to(path)): sha256_file(item)
                for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
            }
        )
    raise FileNotFoundError(path)


def code_tree_hash(package_root: Path) -> str:
    candidates = [
        package_root / "pyproject.toml",
        package_root / "README.md",
        *sorted((package_root / "config").glob("*.yaml")),
        *sorted((package_root / "src").rglob("*.py")),
    ]
    return canonical_hash(
        {
            str(path.relative_to(package_root)): sha256_file(path)
            for path in candidates
            if path.is_file()
        }
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    return f"{prefix}:{sha256_text('|'.join(str(part) for part in parts))[:length]}"


def source_files(data_dir: Path) -> list[Path]:
    tables = [
        data_dir / "sessions.parquet",
        data_dir / "session_logs.parquet",
        data_dir / "conversations.parquet",
        data_dir / "checkpoints.parquet",
        data_dir / "commits.parquet",
    ]
    transcripts = sorted((data_dir / "transcripts").glob("*.jsonl"))
    return tables + transcripts


def snapshot_metadata(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def assert_unchanged(before: dict[str, tuple[int, int]]) -> None:
    after = snapshot_metadata(Path(path) for path in before)
    if before != after:
        changed = sorted(set(before) | set(after))
        raise RuntimeError(f"source dataset changed during the run: {changed}")


def build_manifest(
    *,
    stage: str,
    input_paths: Iterable[Path],
    config: dict[str, Any],
    model_settings: dict[str, Any] | None = None,
    code_version: str | None = None,
) -> RunManifest:
    paths = list(input_paths)
    input_hashes = {str(path): sha256_file(path) for path in paths}
    seed = canonical_hash({"stage": stage, "inputs": input_hashes, "config": config})
    return RunManifest(
        run_id=f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{seed[:12]}",
        stage=stage,
        created_at=datetime.now(UTC),
        input_hashes=input_hashes,
        config_hash=canonical_hash(config),
        code_version=code_version or os.environ.get("ENTIREIO_RETRIEVAL_VERSION"),
        model_settings=model_settings or {},
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl_atomic(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for value in values:
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
    temp.replace(path)
