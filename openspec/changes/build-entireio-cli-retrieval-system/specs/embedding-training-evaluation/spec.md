## ADDED Requirements

### Requirement: Lightweight baseline encoder
The system SHALL use `BAAI/bge-small-en-v1.5` as the initial dense encoder, produce normalized embeddings, and use a consistent asymmetric query instruction during training and inference.

#### Scenario: Baseline embeddings are produced
- **WHEN** the normalized evidence corpus and benchmark queries are encoded
- **THEN** the system produces 384-dimensional normalized vectors using the pinned baseline model revision

### Requirement: CPU-feasible fine-tuning
The system SHALL support multi-positive contrastive fine-tuning within the available CPU-only environment and approximately 7.4 GiB of memory. The initial configuration MUST bound sequence length and batch size and freeze lower encoder layers unless a measured full-tuning experiment fits the resource budget.

#### Scenario: Pilot training is executed
- **WHEN** the initial stratified subset and training partition are available
- **THEN** training completes with bounded resource settings, checkpoints its best validation model, and records runtime and memory configuration

### Requirement: Multi-positive and hard-negative training
The system SHALL train question-to-evidence relationships without treating other known positives from the same topic, checkpoint, or claim group as negatives. It SHALL mine difficult candidates and admit them as hard negatives only when silver judgments identify them as irrelevant.

#### Scenario: Query has several supporting records
- **WHEN** a broad query is supported by multiple evidence units
- **THEN** all known supporting units are represented as positives or excluded from negative sampling

#### Scenario: Similar candidate is irrelevant
- **WHEN** a high-ranking candidate receives a grounded relevance grade of 0
- **THEN** it may be used as a hard negative with its provenance retained

### Requirement: Reproducible baseline comparison
The system SHALL evaluate BM25-only, pretrained dense, fine-tuned dense, hybrid, and hybrid-plus-temporal configurations on unchanged validation and frozen evaluation partitions.

#### Scenario: Experiment finishes
- **WHEN** an evaluation configuration completes
- **THEN** it records nDCG@10, MRR@10, Precision@5, Recall@20 or Recall@50, evidence and claim coverage, answerability performance, and temporal correctness with configuration hashes

### Requirement: Evaluation-set isolation
The system MUST NOT use frozen evaluation queries, relevance judgments, expected answers, or evaluation-only evidence relationships to tune the encoder, mine training negatives, select thresholds, or choose ranking weights.

#### Scenario: Training inputs are assembled
- **WHEN** the fine-tuning dataset is created
- **THEN** no record assigned to the frozen evaluation partition is included

