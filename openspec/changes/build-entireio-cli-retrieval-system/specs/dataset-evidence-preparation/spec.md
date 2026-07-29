## ADDED Requirements

### Requirement: Dataset-only evidence boundary
The system SHALL use only the Parquet tables and transcript files present under the configured `entireio-cli` data directory as factual evidence. It MUST NOT clone the repository, fetch source snapshots, browse the web for repository facts, or use other repository material when preparing evidence, generating labels, retrieving results, or answering questions.

#### Scenario: Evidence preparation uses allowed inputs
- **WHEN** the evidence pipeline is run against `/home/emreerhan/projects/merget-live/entireio-cli`
- **THEN** every factual evidence record is traceable to an included Parquet row or transcript event

#### Scenario: External repository material is unavailable
- **WHEN** a claim would require repository content not represented in the included dataset
- **THEN** the pipeline does not import external content and marks the claim as unsupported or partially supported

### Requirement: Normalized evidence records
The system SHALL produce normalized evidence records for sessions, commits and patches, conversation chunks, checkpoints, and derived multi-session topics. Each record MUST contain a stable evidence identifier, evidence type, source identifiers, timestamp when available, textual content, and retrieval metadata.

#### Scenario: Commit evidence is normalized
- **WHEN** an `ok` commit row contains a patch and related session identifiers
- **THEN** the derived record retains the commit SHA, checkpoint key, session relationship, timestamp, changed paths, and a bounded evidence text

#### Scenario: Missing source fields are handled
- **WHEN** a source row lacks a timestamp, transcript, commit SHA, or optional metadata
- **THEN** the pipeline preserves the record where useful, records the missing value, and does not invent replacement evidence

### Requirement: Citation-addressable evidence
The system SHALL make every retrievable evidence unit citable through dataset-native identifiers such as `session_id`, `turn_id`, `checkpoint_pk`, commit SHA, transcript path, and source-row or event location.

#### Scenario: Evidence citation resolves
- **WHEN** a generated claim cites an evidence identifier
- **THEN** the identifier resolves deterministically to the normalized record and its original dataset source

### Requirement: Sanitized external payloads
The system SHALL remove secrets, author email addresses, unnecessary absolute paths, and unrelated raw tool output before evidence is sent to an external model. Original dataset files MUST remain unchanged.

#### Scenario: Sensitive fields are present
- **WHEN** an evidence packet contains an author email, API-like secret, or local absolute path
- **THEN** the external payload contains a redacted or normalized representation and the original source file is not modified

### Requirement: Reproducible derived artifacts
The system SHALL write derived evidence and run manifests under the `entireio-cli` directory without overwriting source Parquet or transcript files. Run manifests MUST record input hashes, configuration, code version when available, and artifact hashes.

#### Scenario: Pipeline is rerun with unchanged inputs
- **WHEN** the same configuration is applied to unchanged source files
- **THEN** record identifiers, partitions, and deterministic derived content remain stable

