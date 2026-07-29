from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any, Iterable

from .config import PilotConfig
from .models import EvidenceRecord, EvidenceType
from .provenance import stable_id


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _rank(seed: int, *parts: object) -> str:
    return hashlib.sha256(f"{seed}|".encode() + "|".join(map(str, parts)).encode()).hexdigest()


def select_pilot(session_rows: list[dict[str, Any]], config: PilotConfig) -> set[str]:
    if len(session_rows) <= config.size:
        return {str(row["session_id"]) for row in session_rows}

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        created = row.get("created_at")
        month_week = created.strftime("%Y-%m-%U") if isinstance(created, datetime) else "unknown"
        files_count = int(row.get("files_touched_count") or 0)
        size_bin = "large" if files_count >= 10 else "medium" if files_count >= 3 else "small"
        buckets[
            (
                month_week,
                str(row.get("agent") or "unknown"),
                str(row.get("user_id") or "unknown"),
                size_bin,
            )
        ].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: _rank(config.seed, row["session_id"]))

    selected: list[str] = []
    ordered_buckets = sorted(buckets, key=lambda key: _rank(config.seed, *key))
    index = 0
    while len(selected) < config.size:
        added = False
        for key in ordered_buckets:
            rows = buckets[key]
            if index < len(rows):
                selected.append(str(rows[index]["session_id"]))
                added = True
                if len(selected) == config.size:
                    break
        if not added:
            break
        index += 1
    return set(selected)


def assign_partitions(
    records: list[EvidenceRecord],
    config: PilotConfig,
) -> tuple[list[EvidenceRecord], dict[str, Any]]:
    session_ids = sorted({session_id for record in records for session_id in record.session_ids})
    groups = UnionFind(session_ids)
    for record in records:
        if record.evidence_type == EvidenceType.TOPIC:
            continue
        related = record.session_ids
        for session_id in related[1:]:
            groups.union(related[0], session_id)
    checkpoint_sessions: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.evidence_type == EvidenceType.TOPIC:
            continue
        for checkpoint in record.checkpoint_pks:
            checkpoint_sessions[checkpoint].update(record.session_ids)
    for related in checkpoint_sessions.values():
        ordered = sorted(related)
        for session_id in ordered[1:]:
            groups.union(ordered[0], session_id)

    train_before = datetime.fromisoformat(config.train_before.replace("Z", "+00:00"))
    validation_before = datetime.fromisoformat(config.validation_before.replace("Z", "+00:00"))
    group_dates: dict[str, list[datetime]] = defaultdict(list)
    for record in records:
        # Session creation is the stable unit-of-work chronology. Commit dates
        # may describe repository objects created outside the captured session
        # and would otherwise pull old sessions into the frozen future split.
        if record.evidence_type == EvidenceType.SESSION and record.timestamp:
            for session_id in record.session_ids:
                group_dates[groups.find(session_id)].append(record.timestamp)

    group_partition: dict[str, str] = {}
    for session_id in session_ids:
        root = groups.find(session_id)
        dates = group_dates.get(root, [])
        representative = max(dates) if dates else None
        if representative is None or representative < train_before:
            partition = "train"
        elif representative < validation_before:
            partition = "validation"
        else:
            partition = "evaluation"
        group_partition[root] = partition

    updated: list[EvidenceRecord] = []
    by_id = {record.evidence_id: record for record in records}
    base_partitions: dict[str, str] = {}
    for record in records:
        if record.evidence_type == EvidenceType.TOPIC:
            continue
        partitions = {
            group_partition[groups.find(session_id)]
            for session_id in record.session_ids
            if session_id in groups.parent
        }
        partition = max(partitions, key=("train", "validation", "evaluation").index) if partitions else "train"
        item = record.model_copy(update={"partition": partition})
        updated.append(item)
        base_partitions[item.evidence_id] = partition

    # Topic clusters are useful across the project lifecycle, but allowing one
    # cross-partition cluster to union all of its sessions collapses a
    # chronological split. Materialize partition-local topic views instead.
    for record in records:
        if record.evidence_type != EvidenceType.TOPIC:
            continue
        parent_groups: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for parent_id in record.parent_ids:
            partition = base_partitions.get(parent_id)
            parent = by_id.get(parent_id)
            if partition and parent:
                parent_groups[partition].append(parent)
        for partition, parents in sorted(parent_groups.items()):
            session_subset = sorted({value for item in parents for value in item.session_ids})
            if len(session_subset) < 2:
                continue
            lines = [f"Observed work topic: {record.topic_key}"]
            for parent in sorted(
                parents,
                key=lambda item: (
                    item.timestamp or datetime.min.replace(tzinfo=UTC),
                    item.evidence_id,
                ),
            )[-20:]:
                date = parent.timestamp.date().isoformat() if parent.timestamp else "unknown-date"
                lines.append(f"- {date}: {parent.title} [{parent.evidence_id}]")
            updated.append(
                record.model_copy(
                    update={
                        "evidence_id": stable_id(record.evidence_id, partition),
                        "text": "\n".join(lines),
                        "source_refs": [
                            ref for item in parents for ref in item.source_refs
                        ][:100],
                        "parent_ids": [item.evidence_id for item in parents],
                        "session_ids": session_subset,
                        "checkpoint_pks": sorted(
                            {value for item in parents for value in item.checkpoint_pks}
                        ),
                        "commit_shas": sorted(
                            {value for item in parents for value in item.commit_shas}
                        ),
                        "files": sorted({value for item in parents for value in item.files}),
                        "topic_key": f"{record.topic_key}@{partition}",
                        "partition": partition,
                    }
                )
            )

    report = partition_report(updated)
    assert_no_partition_leakage(updated)
    return updated, report


def assert_no_partition_leakage(records: list[EvidenceRecord]) -> None:
    session_partitions: dict[str, set[str]] = defaultdict(set)
    checkpoint_partitions: dict[str, set[str]] = defaultdict(set)
    topic_partitions: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.partition is None:
            raise AssertionError(f"unpartitioned evidence: {record.evidence_id}")
        for session_id in record.session_ids:
            session_partitions[session_id].add(record.partition)
        for checkpoint in record.checkpoint_pks:
            checkpoint_partitions[checkpoint].add(record.partition)
        if record.topic_key:
            topic_partitions[record.topic_key].add(record.partition)
    leaked_sessions = {key: value for key, value in session_partitions.items() if len(value) > 1}
    leaked_checkpoints = {key: value for key, value in checkpoint_partitions.items() if len(value) > 1}
    leaked_topics = {key: value for key, value in topic_partitions.items() if len(value) > 1}
    if leaked_sessions or leaked_checkpoints or leaked_topics:
        raise AssertionError(
            f"partition leakage sessions={len(leaked_sessions)} "
            f"checkpoints={len(leaked_checkpoints)} topics={len(leaked_topics)}"
        )


def partition_report(records: list[EvidenceRecord]) -> dict[str, Any]:
    partitions = Counter(record.partition for record in records)
    sessions: dict[str, set[str]] = defaultdict(set)
    types: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        partition = record.partition or "missing"
        sessions[partition].update(record.session_ids)
        types[partition][record.evidence_type.value] += 1
    return {
        "evidence": dict(partitions),
        "sessions": {key: len(value) for key, value in sessions.items()},
        "types": {key: dict(value) for key, value in types.items()},
    }
