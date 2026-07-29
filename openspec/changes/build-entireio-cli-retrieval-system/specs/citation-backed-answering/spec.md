## ADDED Requirements

### Requirement: Evidence-grounded natural-language answers
The system SHALL generate answers using only evidence returned by the retrieval system and SHALL attach one or more resolving dataset-native citations to every factual claim.

#### Scenario: Supported question is answered
- **WHEN** retrieved evidence adequately supports the query
- **THEN** the answer states the supported conclusions and cites the relevant sessions, turns, checkpoints, commits, or transcript events

### Requirement: Dataset-scoped language
Answers SHALL describe what is observed in the included dataset and MUST NOT imply exhaustive knowledge of the complete or current repository.

#### Scenario: User asks about programming languages
- **WHEN** evidence shows language use in recorded files and patches
- **THEN** the answer describes languages observed in recorded work and qualifies that the result is not necessarily an exhaustive repository inventory

### Requirement: Partial support and abstention
The system SHALL distinguish fully supported, partially supported, and unsupported answers. It MUST abstain from unsupported factual conclusions and explain that adequate evidence was not found in the indexed dataset.

#### Scenario: Question is partially supported
- **WHEN** retrieval supports some expected aspects but not a complete answer
- **THEN** the answer presents only supported claims, identifies the limitation, and is labeled partially supported

#### Scenario: Question is unsupported
- **WHEN** no retrieved evidence passes configured support thresholds
- **THEN** the system returns an `unsupported_by_dataset` response without using general model knowledge to fill the gap

### Requirement: Citation validation
The system SHALL validate that every answer citation resolves and that the cited evidence supports the associated claim before returning or persisting the answer.

#### Scenario: Citation is invalid
- **WHEN** a generated answer cites a missing record or unsupported claim
- **THEN** the answer is regenerated or rejected and the validation failure is recorded

### Requirement: Answer evaluation
The system SHALL evaluate answers against silver expected claims and evidence using claim coverage, citation precision and recall, groundedness, answerability classification, temporal correctness, and unsupported-query false-answer rate.

#### Scenario: Answer experiment completes
- **WHEN** the frozen evaluation query set is answered
- **THEN** aggregate and per-query metrics, answer text, citations, model configuration, and validation outcomes are persisted

