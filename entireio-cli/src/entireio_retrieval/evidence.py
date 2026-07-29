from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import EvidenceConfig
from .io import DatasetReader, build_relationships, parse_json_list
from .models import EvidenceRecord, EvidenceType, SourceRef
from .provenance import sha256_text, stable_id, write_jsonl_atomic
from .security import redact_text

LOW_VALUE_TURN_TYPES = {"progress", "file_snapshot", "system_event", "queue_operation"}
USEFUL_TURN_TYPES = {
    "user_prompt",
    "assistant_response",
    "assistant_thinking",
    "tool_use",
    "tool_result",
    "summary",
}
DIFF_BOUNDARY_RE = re.compile(r"(?=^diff --git )", re.MULTILINE)


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def bounded_chunks(text: str, size: int, overlap: int) -> Iterator[str]:
    value = text.strip()
    if not value:
        return
    if len(value) <= size:
        yield value
        return
    start = 0
    while start < len(value):
        end = min(len(value), start + size)
        if end < len(value):
            boundary = value.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield value[start:end].strip()
        if end >= len(value):
            break
        start = max(start + 1, end - overlap)


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _title(turn_type: str, tool_name: str | None) -> str:
    return f"{turn_type}: {tool_name}" if tool_name else turn_type.replace("_", " ")


def _record(
    *,
    evidence_type: EvidenceType,
    title: str,
    text: str,
    source_refs: list[SourceRef],
    timestamp: datetime | None = None,
    parent_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    checkpoint_pks: list[str] | None = None,
    commit_shas: list[str] | None = None,
    files: list[str] | None = None,
    branch: str | None = None,
    agent: str | None = None,
    topic_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    identity: tuple[object, ...],
) -> EvidenceRecord:
    # Redaction can expand repeated short secrets/paths into longer placeholders,
    # so enforce the evidence bound after sanitization as the final step.
    clean = redact_text(text).strip()[:8000]
    digest = sha256_text(clean)
    return EvidenceRecord(
        evidence_id=stable_id(evidence_type.value, *identity, digest),
        evidence_type=evidence_type,
        title=redact_text(title)[:500],
        text=clean,
        timestamp=timestamp,
        source_refs=source_refs,
        parent_ids=parent_ids or [],
        session_ids=session_ids or [],
        checkpoint_pks=checkpoint_pks or [],
        commit_shas=commit_shas or [],
        files=files or [],
        branch=branch or None,
        agent=agent or None,
        topic_key=topic_key,
        metadata=metadata or {},
        content_hash=f"sha256:{digest}",
    )


class EvidenceBuilder:
    def __init__(self, reader: DatasetReader, config: EvidenceConfig):
        self.reader = reader
        self.config = config

    def build(self, session_ids: set[str] | None = None) -> list[EvidenceRecord]:
        relationships = build_relationships(self.reader)
        conversations = self._conversation_evidence(session_ids)
        transcripts = self._transcript_evidence(relationships, session_ids)
        checkpoints = self._checkpoint_evidence(relationships, session_ids)
        commits = self._commit_evidence(relationships, session_ids)
        sessions = self._session_evidence(
            relationships, conversations, transcripts, checkpoints, commits, session_ids
        )
        topics = self._topic_evidence(sessions, commits)
        records = conversations + transcripts + checkpoints + commits + sessions + topics
        records.sort(key=lambda item: (item.timestamp or datetime.min.replace(tzinfo=UTC), item.evidence_id))
        return records

    def _transcript_evidence(
        self,
        relationships: dict[str, Any],
        session_ids: set[str] | None,
    ) -> list[EvidenceRecord]:
        output: list[EvidenceRecord] = []
        for session_id, log in relationships["session_logs"].items():
            if session_ids is not None and session_id not in session_ids:
                continue
            transcript_path = str(log.get("transcript_path") or "")
            if not transcript_path:
                continue
            for event_index, event in self.reader.iter_transcript(transcript_path):
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type") or "")
                if event_type not in {"user", "assistant"}:
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                parts: list[str] = []
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                            continue
                        if not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type") or "")
                        if block_type == "text" and block.get("text"):
                            parts.append(str(block["text"]))
                        elif (
                            block_type == "thinking"
                            and self.config.include_thinking
                            and block.get("thinking")
                        ):
                            parts.append(str(block["thinking"]))
                        elif block_type == "tool_use":
                            tool_name = str(block.get("name") or "unknown")
                            tool_input = json.dumps(
                                block.get("input", {}),
                                sort_keys=True,
                                default=str,
                            )
                            parts.append(f"Tool: {tool_name}\nInput: {tool_input}")
                        elif (
                            block_type == "tool_result"
                            and self.config.include_tool_results
                            and block.get("content")
                        ):
                            parts.append(f"Tool result:\n{block['content']}")
                body = "\n\n".join(part for part in parts if part.strip())
                for chunk_index, chunk in enumerate(
                    bounded_chunks(
                        body,
                        self.config.chunk_chars,
                        self.config.chunk_overlap,
                    )
                ):
                    output.append(
                        _record(
                            evidence_type=EvidenceType.TRANSCRIPT,
                            title=f"Transcript {event_type} event",
                            text=chunk,
                            timestamp=_timestamp(event.get("timestamp")),
                            source_refs=[
                                SourceRef(
                                    source_file=transcript_path,
                                    source_kind="transcript",
                                    event_index=event_index,
                                    session_id=session_id,
                                    transcript_path=transcript_path,
                                )
                            ],
                            session_ids=[session_id],
                            branch=event.get("gitBranch"),
                            metadata={
                                "event_type": event_type,
                                "message_role": message.get("role"),
                                "chunk_index": chunk_index,
                            },
                            identity=(
                                transcript_path,
                                event_index,
                                chunk_index,
                            ),
                        )
                    )
        return output

    def _checkpoint_evidence(
        self,
        relationships: dict[str, Any],
        session_ids: set[str] | None,
    ) -> list[EvidenceRecord]:
        output: list[EvidenceRecord] = []
        for checkpoint_pk, checkpoint in relationships["checkpoints"].items():
            linked_sessions = checkpoint.get("session_ids_parsed", [])
            selected_sessions = [
                session_id
                for session_id in linked_sessions
                if session_ids is None or session_id in session_ids
            ]
            if session_ids is not None and not selected_sessions:
                continue
            commits = checkpoint.get("commit_shas_parsed", [])
            files = parse_json_list(checkpoint.get("files_touched"))
            lines = [
                f"Checkpoint: {checkpoint_pk}",
                f"Strategy: {checkpoint.get('strategy') or 'unknown'}",
                f"Branch: {checkpoint.get('branch') or 'unknown'}",
                f"Linked sessions: {len(selected_sessions)}",
                f"Recorded commits: {len(commits)}",
                f"Recorded additions: {checkpoint.get('total_additions') or 0}",
                f"Recorded deletions: {checkpoint.get('total_deletions') or 0}",
            ]
            if files:
                lines.append("Files involved:\n- " + "\n- ".join(files[:100]))
            output.append(
                _record(
                    evidence_type=EvidenceType.CHECKPOINT,
                    title=f"Checkpoint {checkpoint_pk}",
                    text="\n".join(lines),
                    source_refs=[
                        SourceRef(
                            source_file="checkpoints.parquet",
                            source_kind="parquet",
                            source_row=int(checkpoint["__source_row__"]),
                            checkpoint_pk=checkpoint_pk,
                        )
                    ],
                    session_ids=selected_sessions,
                    checkpoint_pks=[checkpoint_pk],
                    commit_shas=commits,
                    files=files,
                    branch=checkpoint.get("branch"),
                    metadata={
                        "checkpoint_id": checkpoint.get("checkpoint_id"),
                        "timestamp_missing": True,
                    },
                    identity=(checkpoint_pk,),
                )
            )
        return output

    def _conversation_evidence(self, session_ids: set[str] | None) -> list[EvidenceRecord]:
        output: list[EvidenceRecord] = []
        columns = [
            "turn_id", "session_id", "checkpoint_pk", "turn_type", "content",
            "timestamp", "tool_name", "file_path", "command", "pattern",
            "branch", "agent", "category",
        ]
        available = set(self.reader.read_table("conversations").column_names)
        columns = [column for column in columns if column in available]
        for row_number, row in self.reader.iter_table("conversations", columns=columns):
            session_id = str(row["session_id"])
            if session_ids is not None and session_id not in session_ids:
                continue
            turn_type = str(row.get("turn_type") or "")
            if turn_type in LOW_VALUE_TURN_TYPES or turn_type not in USEFUL_TURN_TYPES:
                continue
            if turn_type == "assistant_thinking" and not self.config.include_thinking:
                continue
            if turn_type == "tool_result" and not self.config.include_tool_results:
                continue
            content = str(row.get("content") or "")
            if not content.strip():
                continue
            prefix: list[str] = []
            for label, key in (("Tool", "tool_name"), ("File", "file_path"), ("Command", "command"), ("Pattern", "pattern")):
                if row.get(key):
                    prefix.append(f"{label}: {row[key]}")
            body = "\n".join(prefix + [content])
            for chunk_index, chunk in enumerate(
                bounded_chunks(body, self.config.chunk_chars, self.config.chunk_overlap)
            ):
                file_path = row.get("file_path")
                output.append(
                    _record(
                        evidence_type=EvidenceType.CONVERSATION,
                        title=_title(turn_type, row.get("tool_name")),
                        text=chunk,
                        timestamp=_timestamp(row.get("timestamp")),
                        source_refs=[
                            SourceRef(
                                source_file="conversations.parquet",
                                source_kind="parquet",
                                source_row=row_number,
                                session_id=session_id,
                                turn_id=row.get("turn_id"),
                                checkpoint_pk=row.get("checkpoint_pk"),
                            )
                        ],
                        session_ids=[session_id],
                        checkpoint_pks=_unique([row.get("checkpoint_pk")]),
                        files=_unique([str(file_path) if file_path else None]),
                        branch=row.get("branch"),
                        agent=row.get("agent"),
                        metadata={
                            "turn_type": turn_type,
                            "tool_name": row.get("tool_name"),
                            "category": row.get("category"),
                            "chunk_index": chunk_index,
                        },
                        identity=(row_number, row.get("turn_id"), chunk_index),
                    )
                )
        return output

    def _commit_evidence(
        self, relationships: dict[str, Any], session_ids: set[str] | None
    ) -> list[EvidenceRecord]:
        output: list[EvidenceRecord] = []
        checkpoints = relationships["checkpoints"]
        for checkpoint_pk, commits in relationships["commits_by_checkpoint"].items():
            checkpoint = checkpoints.get(checkpoint_pk, {})
            linked_sessions = checkpoint.get("session_ids_parsed", [])
            selected_sessions = [
                session_id for session_id in linked_sessions
                if session_ids is None or session_id in session_ids
            ]
            if session_ids is not None and not selected_sessions:
                continue
            for commit in commits:
                if commit.get("status") != "ok" or not commit.get("commit_sha"):
                    continue
                message = str(commit.get("commit_message") or "").strip()
                patch = str(commit.get("patch") or "").strip()
                files = []
                raw_files = str(commit.get("files_changed") or "")
                for line in raw_files.splitlines():
                    if "\t" in line:
                        files.append(line.rsplit("\t", 1)[-1])
                sections = [section.strip() for section in DIFF_BOUNDARY_RE.split(patch) if section.strip()]
                if not sections:
                    sections = [patch] if patch else [message]
                for section_index, section in enumerate(sections):
                    body = f"Commit message:\n{message}\n\nChange:\n{section}".strip()
                    for chunk_index, chunk in enumerate(
                        bounded_chunks(body, self.config.chunk_chars, self.config.chunk_overlap)
                    ):
                        output.append(
                            _record(
                                evidence_type=EvidenceType.COMMIT,
                                title=message.splitlines()[0] if message else f"Commit {commit['commit_sha'][:12]}",
                                text=chunk,
                                timestamp=_timestamp(commit.get("commit_date") or commit.get("author_date")),
                                source_refs=[
                                    SourceRef(
                                        source_file="commits.parquet",
                                        source_kind="parquet",
                                        source_row=commit["__source_row__"],
                                        checkpoint_pk=checkpoint_pk,
                                        commit_sha=commit["commit_sha"],
                                    )
                                ],
                                session_ids=selected_sessions,
                                checkpoint_pks=[checkpoint_pk],
                                commit_shas=[commit["commit_sha"]],
                                files=_unique(files),
                                branch=commit.get("branch"),
                                metadata={
                                    "section_index": section_index,
                                    "chunk_index": chunk_index,
                                    "additions": commit.get("total_additions") or 0,
                                    "deletions": commit.get("total_deletions") or 0,
                                },
                                identity=(
                                    commit["__source_row__"],
                                    commit["commit_sha"],
                                    section_index,
                                    chunk_index,
                                ),
                            )
                        )
        return output

    def _session_evidence(
        self,
        relationships: dict[str, Any],
        conversations: list[EvidenceRecord],
        transcripts: list[EvidenceRecord],
        checkpoints: list[EvidenceRecord],
        commits: list[EvidenceRecord],
        session_ids: set[str] | None,
    ) -> list[EvidenceRecord]:
        conversation_by_session: dict[str, list[EvidenceRecord]] = defaultdict(list)
        commit_by_session: dict[str, list[EvidenceRecord]] = defaultdict(list)
        transcript_by_session: dict[str, list[EvidenceRecord]] = defaultdict(list)
        checkpoint_by_session: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in conversations:
            for session_id in record.session_ids:
                conversation_by_session[session_id].append(record)
        for record in commits:
            for session_id in record.session_ids:
                commit_by_session[session_id].append(record)
        for record in transcripts:
            for session_id in record.session_ids:
                transcript_by_session[session_id].append(record)
        for record in checkpoints:
            for session_id in record.session_ids:
                checkpoint_by_session[session_id].append(record)

        output: list[EvidenceRecord] = []
        for session_id, session in relationships["sessions"].items():
            if session_ids is not None and session_id not in session_ids:
                continue
            conv = conversation_by_session.get(session_id, [])
            related_commits = commit_by_session.get(session_id, [])
            user_prompts = [
                item.text for item in conv
                if item.metadata.get("turn_type") == "user_prompt"
            ][:8]
            conclusions = [
                item.text for item in conv
                if item.metadata.get("turn_type") == "assistant_response"
            ][-3:]
            commit_titles = _unique(item.title for item in related_commits)
            files = _unique(
                parse_json_list(session.get("files_touched"))
                + [path for item in related_commits for path in item.files]
            )
            sections = [
                f"Session: {session_id}",
                f"Agent: {session.get('agent') or 'unknown'}",
                f"Strategy: {session.get('strategy') or 'unknown'}",
            ]
            if user_prompts:
                sections.append("Problems and requests:\n- " + "\n- ".join(user_prompts))
            if commit_titles:
                sections.append("Recorded commit outcomes:\n- " + "\n- ".join(commit_titles))
            if conclusions:
                sections.append("Assistant conclusions:\n- " + "\n- ".join(conclusions))
            if files:
                sections.append("Files involved:\n- " + "\n- ".join(files[:100]))
            if session.get("missing"):
                sections.append("Missing source relationships: " + ", ".join(session["missing"]))
            text = "\n\n".join(sections)
            source_refs = [
                SourceRef(
                    source_file="sessions.parquet",
                    source_kind="parquet",
                    source_row=int(session["__source_row__"]),
                    session_id=session_id,
                )
            ]
            log = session.get("session_log")
            if log and log.get("transcript_path"):
                source_refs.append(
                    SourceRef(
                        source_file="session_logs.parquet",
                        source_kind="parquet",
                        source_row=int(log["__source_row__"]),
                        session_id=session_id,
                        transcript_path=log["transcript_path"],
                    )
                )
            output.append(
                _record(
                    evidence_type=EvidenceType.SESSION,
                    title=f"Session {session_id}",
                    text=text[: self.config.max_chars],
                    timestamp=_timestamp(session.get("created_at")),
                    source_refs=source_refs,
                    parent_ids=_unique(
                        [item.evidence_id for item in conv]
                        + [
                            item.evidence_id
                            for item in transcript_by_session.get(session_id, [])
                        ]
                        + [
                            item.evidence_id
                            for item in checkpoint_by_session.get(session_id, [])
                        ]
                        + [item.evidence_id for item in related_commits]
                    ),
                    session_ids=[session_id],
                    checkpoint_pks=session.get("checkpoint_ids_parsed", []),
                    commit_shas=_unique(item for record in related_commits for item in record.commit_shas),
                    files=files,
                    branch=session.get("branch"),
                    agent=session.get("agent"),
                    metadata={
                        "prompt_count": session.get("prompt_count") or 0,
                        "tool_call_count": session.get("tool_call_count") or 0,
                        "session_success": session.get("session_success"),
                        "missing": session.get("missing", []),
                    },
                    identity=(session_id,),
                )
            )
        return output

    def _topic_evidence(
        self, sessions: list[EvidenceRecord], commits: list[EvidenceRecord]
    ) -> list[EvidenceRecord]:
        topic_members: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in sessions + commits:
            keys = set()
            for path in record.files:
                parts = [part for part in path.split("/") if part]
                if len(parts) >= 3 and parts[0] in {"cmd", "internal", "pkg"}:
                    keys.add("/".join(parts[:3]))
                elif len(parts) >= 2:
                    keys.add("/".join(parts[:2]))
                elif parts:
                    keys.add(parts[0])
            for key in sorted(keys)[:5]:
                topic_members[key].append(record)

        output: list[EvidenceRecord] = []
        for topic_key, members in sorted(topic_members.items()):
            session_set = _unique(item for member in members for item in member.session_ids)
            if len(session_set) < 2:
                continue
            ranked = sorted(
                members,
                key=lambda item: (item.timestamp or datetime.min.replace(tzinfo=UTC), item.evidence_id),
            )
            lines = [f"Observed work topic: {topic_key}"]
            for member in ranked[-20:]:
                date = member.timestamp.date().isoformat() if member.timestamp else "unknown-date"
                lines.append(f"- {date}: {member.title} [{member.evidence_id}]")
            files = Counter(path for member in members for path in member.files)
            if files:
                lines.append("Frequently involved files:")
                lines.extend(f"- {path}" for path, _ in files.most_common(20))
            output.append(
                _record(
                    evidence_type=EvidenceType.TOPIC,
                    title=f"Topic: {topic_key}",
                    text="\n".join(lines),
                    timestamp=max((item.timestamp for item in members if item.timestamp), default=None),
                    source_refs=list(
                        {
                            (
                                ref.source_file,
                                ref.source_row,
                                ref.event_index,
                                ref.session_id,
                                ref.turn_id,
                                ref.checkpoint_pk,
                                ref.commit_sha,
                            ): ref
                            for member in members
                            for ref in member.source_refs
                        }.values()
                    )[:100],
                    parent_ids=_unique(member.evidence_id for member in members),
                    session_ids=session_set,
                    checkpoint_pks=_unique(item for member in members for item in member.checkpoint_pks),
                    commit_shas=_unique(item for member in members for item in member.commit_shas),
                    files=[path for path, _ in files.most_common(100)],
                    topic_key=topic_key,
                    metadata={"member_count": len(members), "session_count": len(session_set)},
                    identity=(topic_key, *sorted(member.evidence_id for member in members)),
                )
            )
        return output


def write_evidence(path: Path, records: Iterable[EvidenceRecord]) -> None:
    write_jsonl_atomic(path, records)


def load_evidence(path: Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(EvidenceRecord.model_validate_json(line))
    return records


def evidence_quality(records: list[EvidenceRecord]) -> dict[str, Any]:
    by_type = Counter(record.evidence_type.value for record in records)
    unresolved = sum(not record.source_refs for record in records)
    duplicate_ids = len(records) - len({record.evidence_id for record in records})
    empty = sum(not record.text.strip() for record in records)
    return {
        "total": len(records),
        "by_type": dict(sorted(by_type.items())),
        "unresolved_sources": unresolved,
        "duplicate_ids": duplicate_ids,
        "empty_text": empty,
        "max_chars": max((len(record.text) for record in records), default=0),
    }
