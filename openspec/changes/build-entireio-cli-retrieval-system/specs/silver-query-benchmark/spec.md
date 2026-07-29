## ADDED Requirements

### Requirement: Repository-understanding query generation
The system SHALL generate natural questions about work observed in the repository dataset across technology, architecture, behavior, rationale, history, testing, implementation, and unsupported-query categories. Generated questions MUST include repository-, subsystem-, and change-level scopes.

#### Scenario: Broad supported question is generated
- **WHEN** an evidence cluster supports a repository-level observation such as languages appearing in recorded work
- **THEN** the system generates a standalone natural-language question with expected claims and all supporting evidence identifiers

#### Scenario: Unsupported question is generated
- **WHEN** a repository-understanding question has no adequate evidence in the dataset
- **THEN** the query is labeled `unsupported_by_dataset` and contains no fabricated positive evidence

### Requirement: Structured Luna generation
The system SHALL use OpenRouter model `openai/gpt-5.6-luna` with medium reasoning and strict structured output for query generation. The API key MUST be read at runtime from the configured key file and MUST NOT appear in logs or artifacts.

#### Scenario: Query-generation response succeeds
- **WHEN** Luna returns a response conforming to the required schema
- **THEN** the system persists the query, scope, intent, temporal mode, answerability, expected claims, evidence identifiers, and generation provenance

#### Scenario: Response is invalid or transiently fails
- **WHEN** Luna returns invalid structured output or a retryable API error
- **THEN** the system performs bounded retries and records a non-secret failure result if generation cannot complete

### Requirement: Three-state answerability
Every generated query SHALL be labeled `supported`, `partially_supported`, or `unsupported_by_dataset`. Claims about missing evidence MUST be phrased as absence from the indexed dataset rather than absence from the repository.

#### Scenario: Evidence is incomplete
- **WHEN** the dataset supports part but not all of a broad question
- **THEN** the record is labeled `partially_supported` and distinguishes supported claims from unresolved aspects

### Requirement: Independent automated validation
The system SHALL validate generated queries with a separate Luna critic call and deterministic checks for grounding, evidence existence, answerability, evidence sufficiency, ambiguity, naturalness, source-language leakage, and temporal correctness.

#### Scenario: Critic and deterministic checks pass
- **WHEN** a generated query meets configured acceptance thresholds and every cited source resolves
- **THEN** the query is admitted to the silver benchmark

#### Scenario: Unsupported claim is detected
- **WHEN** a claim lacks a resolving evidence identifier or contradicts its cited evidence
- **THEN** the query is rejected with a recorded reason

### Requirement: Leakage-resistant splitting and deduplication
The system SHALL assign evidence groups to training, validation, or evaluation before query generation. Queries sharing a source session, checkpoint group, evidence topic, or paraphrase family MUST NOT cross partitions, and exact or semantic duplicates MUST be removed.

#### Scenario: Related queries are generated
- **WHEN** multiple queries are derived from the same evidence cluster
- **THEN** all such queries remain in the cluster's preassigned partition

### Requirement: Silver relevance judgments
The system SHALL build pooled candidates from independent retrieval and metadata methods and use an evidence-grounded Luna judging pass to assign relevance grades from 0 through 3. Nonzero judgments MUST identify supported expected claims and resolving evidence.

#### Scenario: Candidate directly supports an answer
- **WHEN** a candidate provides direct evidence for one or more expected claims
- **THEN** the judge may assign grade 3 and records the supported claims and citations

#### Scenario: Candidate order is evaluated
- **WHEN** candidates are sent for automated judging
- **THEN** their order is randomized and judgment provenance is stored to reduce systematic ordering bias

### Requirement: Benchmark provenance
The benchmark SHALL record source hashes, evidence partition, model slug, reasoning effort, generation and critic prompt versions, sampling configuration, raw non-secret responses, validation outcomes, and artifact hashes.

#### Scenario: Benchmark result is inspected
- **WHEN** a query or relevance label is reviewed after generation
- **THEN** its complete generation and validation lineage can be reconstructed without exposing the API key

