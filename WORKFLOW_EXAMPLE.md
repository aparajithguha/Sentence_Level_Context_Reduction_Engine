# 🔬 SCRE Workflow Example
## Same Document. Same Query. Four Very Different Answers.

This document walks through a realistic, concrete example. We take one technical decision document, ask one question, and show exactly what each retrieval strategy returns — and why.

---

## 📄 The Source Document

A fictional but realistic Architecture Decision Record (ADR) for a distributed system:

```
Document ID: "adr-database-selection"
Length:      ~450 tokens
```

```
# ADR-007: Database Selection for Event Processing Service

## Background

The Event Processing Service needs a persistent store for incoming telemetry events
from 400+ edge devices. Events arrive at ~12,000 per second at peak load.

## Goals

The selected database must handle high write throughput without sacrificing read
latency for dashboards. It must support horizontal scaling as device count grows.

## Non-Goals

We are not building a data warehouse. Long-term analytics will be handled by a
separate pipeline. This decision covers only the hot-path event store.

## Options Considered

We evaluated three candidates: PostgreSQL, Apache Cassandra, and Redis Streams.

PostgreSQL was attractive due to developer familiarity and strong ACID guarantees.
However, under write-heavy load testing it saturated at 4,200 writes/sec on our
target hardware, which is well below the 12,000/sec peak requirement.

Apache Cassandra was evaluated for its linear horizontal scaling and
write-optimized LSM-tree storage engine. It achieved 18,500 writes/sec in our
benchmark, exceeding the peak requirement by 54%. Read latency for point lookups
was 3.2ms at P99, which is acceptable for dashboard refresh at 5-second intervals.

Redis Streams was fast (>50,000 writes/sec) but offers no durability guarantees
without AOF persistence, which halves throughput. It also lacks the query
flexibility needed for ad-hoc dashboard filtering.

## Decision

We selected Apache Cassandra as the event store because it is the only candidate
that satisfies both the write throughput requirement (12,000/sec) and the read
latency requirement (P99 < 10ms) simultaneously.

## Constraints

All writes must use the LOCAL_QUORUM consistency level to prevent data loss during
a single datacenter outage. Read operations may use LOCAL_ONE for lower latency.
Cassandra cluster must have a minimum of 3 nodes per datacenter.

## Implementation Plan

1. Provision a 3-node Cassandra cluster using Terraform in us-east-1.
2. Define the keyspace with NetworkTopologyStrategy and replication factor 3.
3. Create the events table with a composite partition key (device_id, date_bucket).
4. Implement the writer service using the DataStax Java driver with async batching.
5. Set up Prometheus JMX exporter for Cassandra metrics collection.
6. Validate throughput under simulated peak load before promoting to production.

## Alternatives Rejected

PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required).
Redis Streams: No durable persistence at required throughput without 50% penalty.

## Risks

If device count grows beyond 3× current projections, the cluster will require
re-partitioning. This is a known operational cost of Cassandra at scale.
```

---

## ❓ The Query

```
"Why was Apache Cassandra selected, and what are the write constraints?"
```

This query has two parts:
- A **decision trace** request — *why* was something chosen
- A **constraint extraction** request — *what rules* govern the implementation

The ideal answer needs to pull together: the decision sentence, the reasons that drove it (performance numbers), and the operational constraints. These are spread across **four different sections** of the document.

---

## Strategy 1 — Raw Context (Baseline)

### What it does
Returns the entire document, unchanged.

### Output
```
# ADR-007: Database Selection for Event Processing Service
[... all 450 tokens ...]
```

### Analysis

| Metric | Value |
|--------|-------|
| Tokens sent to LLM | ~450 |
| Contains the answer | ✅ Yes |
| Contains noise | ✅ Yes — Background, Goals, Non-Goals, Implementation Plan all irrelevant to this query |
| Cost | 100% |

> **The ceiling.** Raw context always has the answer. But for long documents (10,000+ tokens), it becomes the most expensive and least focused option. The LLM must sift through everything to find what matters.

---

## Strategy 2 — BM25 Keyword Retrieval

### What it does
Tokenises the query into terms: `["apache", "cassandra", "selected", "write", "constraints"]`. Scores every sentence by BM25Okapi term frequency weighted against inverse document frequency. Returns the top-6 highest-scoring sentences.

### Step-by-step

**Query terms scored:**

| Term | IDF weight (approx.) | Reason |
|------|---------------------|--------|
| `cassandra` | High | Appears in ~5 sentences out of 20 |
| `write` | Medium | Appears in ~7 sentences |
| `selected` | Medium | Appears in ~3 sentences |
| `apache` | High | Appears in ~4 sentences |
| `constraints` | High | Appears in ~2 sentences (section header) |

**Top-6 sentences by BM25 score:**

```
1. "Apache Cassandra was evaluated for its linear horizontal scaling and
    write-optimized LSM-tree storage engine."

2. "We selected Apache Cassandra as the event store because it is the only
    candidate that satisfies both the write throughput requirement (12,000/sec)..."

3. "All writes must use the LOCAL_QUORUM consistency level to prevent data
    loss during a single datacenter outage."

4. "Redis Streams was fast (>50,000 writes/sec) but offers no durability
    guarantees without AOF persistence, which halves throughput."

5. "We evaluated three candidates: PostgreSQL, Apache Cassandra, and Redis Streams."

6. "PostgreSQL was attractive due to developer familiarity and strong ACID
    guarantees."
```

### Analysis

| Metric | Value |
|--------|-------|
| Tokens sent to LLM | ~98 |
| Contains the decision | ✅ Sentence 2 |
| Contains the constraints | ✅ Sentence 3 |
| Contains noise | ⚠️ Sentences 4–6 are about *rejected* alternatives — not the answer |
| Missing | ❌ Performance numbers that *justify* the decision (18,500/sec, 3.2ms P99) |
| Missing | ❌ Cassandra-specific constraints (min 3 nodes, replication factor) |

> **The problem:** BM25 retrieves sentences that *mention* the right words. It cannot distinguish between "Cassandra was evaluated" (background) and "Cassandra was selected" (the actual decision). It has no concept of the reasoning chain that links the throughput benchmark → the decision → the constraint.

---

## Strategy 3 — Vector Similarity Search

### What it does
Encodes the entire query into a dense embedding vector using `all-mpnet-base-v2`. Encodes every sentence independently. Computes cosine similarity between the query vector and each sentence vector. Returns the top-6 closest sentences.

### Step-by-step

**Query embedding (conceptual):**
```
query_vec = encode("Why was Apache Cassandra selected, and what are the write constraints?")
# Semantically close to: decisions, justifications, performance requirements, constraints
```

**Top cosine similarities:**

| Score | Sentence |
|-------|----------|
| 0.91 | *"We selected Apache Cassandra as the event store because it is the only candidate..."* |
| 0.84 | *"All writes must use LOCAL_QUORUM consistency level to prevent data loss..."* |
| 0.81 | *"Apache Cassandra was evaluated for its linear horizontal scaling and write-optimized LSM-tree..."* |
| 0.76 | *"Redis Streams was fast (>50,000 writes/sec) but offers no durability guarantees..."* |
| 0.74 | *"PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required)."* |
| 0.71 | *"Read operations may use LOCAL_ONE for lower latency."* |

**Output (6 sentences, ~75 tokens):**
```
We selected Apache Cassandra as the event store because it is the only candidate
that satisfies both the write throughput requirement (12,000/sec) and the read
latency requirement (P99 < 10ms) simultaneously.

All writes must use the LOCAL_QUORUM consistency level to prevent data loss during
a single datacenter outage.

Apache Cassandra was evaluated for its linear horizontal scaling and write-optimized
LSM-tree storage engine.

Redis Streams was fast (>50,000 writes/sec) but offers no durability guarantees
without AOF persistence, which halves throughput.

PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required).

Read operations may use LOCAL_ONE for lower latency.
```

### Analysis

| Metric | Value |
|--------|-------|
| Tokens sent to LLM | ~75 |
| Contains the decision | ✅ Yes |
| Contains write constraint | ✅ LOCAL_QUORUM sentence |
| Missing | ❌ The **actual performance evidence** — 18,500/sec, P99 3.2ms |
| Missing | ❌ Minimum 3 nodes constraint |
| Noise | ⚠️ Redis and PostgreSQL rejection sentences are included |

> **Better than BM25, but still blind.** Vector search understands meaning — it correctly scores the decision sentence highest. But it treats every sentence as an isolated island. The sentence *"It achieved 18,500 writes/sec in our benchmark, exceeding the peak requirement by 54%"* isn't retrieved even though it's the **critical justification** for the decision. The word "it" (coreference) and the lack of "Cassandra" or "select" in that sentence makes it score lower despite being causally the most important supporting sentence.

---

## Strategy 4 — SCRE

### What it does
SCRE runs six stages. Here is what actually happens internally for this document.

---

### Stage 1 — Parse & Segment

SCRE splits on paragraphs and sentences. It detects the ordered list in "Implementation Plan" and collapses steps 1–6 into a single **WorkflowUnit**:

```python
WorkflowUnit(
  name="Implementation Plan",
  steps=[
    "1. Provision a 3-node Cassandra cluster...",
    "2. Define the keyspace with NetworkTopologyStrategy...",
    "3. Create the events table with composite partition key...",
    "4. Implement the writer service with async batching...",
    "5. Set up Prometheus JMX exporter...",
    "6. Validate throughput under simulated peak load..."
  ]
)
```

This keeps the procedure intact as one atomic unit rather than 6 disconnected sentences.

---

### Stage 2 — Semantic Classification

Every sentence is classified. Here are the key ones:

| # | Sentence (abbreviated) | Type | Why |
|---|------------------------|------|-----|
| 5 | "The selected database must handle high write throughput..." | `constraint` | "must" trigger |
| 8 | "PostgreSQL was attractive due to developer familiarity..." | `fact` | NLP fallback |
| 9 | "However, under write-heavy load testing it saturated at 4,200..." | `outcome` | "however", "saturated" |
| 10 | "Apache Cassandra was evaluated for its linear horizontal scaling..." | `fact` | NLP fallback |
| 11 | "It achieved 18,500 writes/sec in our benchmark, exceeding peak by 54%." | `outcome` | "achieved", "exceeding" |
| 12 | "Read latency for point lookups was 3.2ms at P99..." | `fact` | NLP fallback |
| 15 | "We selected Apache Cassandra as the event store because..." | `decision` | "selected" keyword |
| 17 | "All writes must use LOCAL_QUORUM consistency level..." | `constraint` | "must" trigger |
| 18 | "Read operations may use LOCAL_ONE for lower latency." | `constraint` | dependency pattern |
| 19 | "Cassandra cluster must have a minimum of 3 nodes..." | `constraint` | "must" trigger |
| 20 | WorkflowUnit: Implementation Plan (6 steps) | `workflow` | Ordered list detection |
| 22 | "PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required)." | `decision` | Alternatives Rejected |
| 24 | "If device count grows beyond 3× current projections..." | `outcome` | Risk/consequence |

---

### Stage 3 — Dual Graph Construction

**Knowledge Graph** (entity → sentence indices):

```
"cassandra"       → {10, 11, 15, 17, 18, 19, 20}
"write"           → {5, 9, 11, 17}
"throughput"      → {5, 9, 11, 22}
"postgresql"      → {8, 9, 22}
"local_quorum"    → {17}
"latency"         → {5, 12, 18}
```

**Reasoning Graph** (causal edges):

```
15 (decision: Cassandra selected)
  ← 11  (outcome:  18,500 writes/sec achieved)   # "because it satisfies"
  ← 12  (fact:     P99 3.2ms latency)             # adjacent causal
  ← 5   (constraint: must handle throughput)      # decision satisfies constraint
  → 17  (constraint: LOCAL_QUORUM required)       # decision drives constraint
  → 19  (constraint: min 3 nodes)                 # decision drives constraint

9 (outcome: PostgreSQL saturated at 4,200)
  → 22  (decision: PostgreSQL rejected)           # evidence → rejection
```

---

### Stage 4 — Query-Aware Hybrid Scoring

Query: `"Why was Apache Cassandra selected, and what are the write constraints?"`

Extracted features:
- **Terms:** `cassandra, apache, selected, write, constraints, event, store`
- **Hard identifiers:** `Cassandra, LOCAL_QUORUM` (alphanumeric casing detected)
- **Intent:** decision-trace + constraint-extraction
- **Noun chunks:** `apache cassandra`, `write constraints`, `event store`

Scores for key sentences:

| # | Type | Base×2 | Lexical | Dense | Entity | Intent | Hard ID | **Total** |
|---|------|--------|---------|-------|--------|--------|---------|-----------|
| 15 | decision | 2.0 | 8.4 | 9.2 | 6.0 | 3.0 | 15.0 | **45.6** ← top |
| 17 | constraint | 2.0 | 5.1 | 7.8 | 3.0 | 3.0 | 15.0 | **38.4** |
| 19 | constraint | 2.0 | 3.2 | 6.1 | 3.0 | 3.0 | 15.0 | **34.8** |
| 11 | outcome | 1.7 | 6.3 | 5.9 | 3.0 | 0.0 | 0.0 | **18.4** |
| 5  | constraint | 2.0 | 4.8 | 5.5 | 0.0 | 3.0 | 0.0 | **17.8** |
| 12 | fact | 1.4 | 3.1 | 4.2 | 0.0 | 0.0 | 0.0 | **9.9** |
| 20 | workflow | 1.9 | 1.2 | 2.1 | 2.0 | 0.0 | 15.0 | **23.2** |

**Top-4 selected by score:** sentences `15, 17, 19, 11`

---

### Stage 5 — Context Expansion

**Adjacent window (±1):**
- Sentence 15 → pulls in 14, 16
- Sentence 11 → pulls in 10, 12

**Reasoning graph (2-hop from selected set `{15, 17, 19, 11}`):**
```
hop1 from 15 → {5, 11, 12, 17, 19}   ← already covered
hop1 from 11 → {9, 12, 15}           ← adds 9
hop2 from 9  → {22}                  ← adds 22 (PostgreSQL rejection)
```

**Final expanded set:** `{5, 9, 10, 11, 12, 14, 15, 16, 17, 19, 22}`

After de-duplication (Jaccard > 0.85 threshold) and re-ordering by original index:

---

### Stage 6 — Final Output

```
[ADR-007: Database Selection for Event Processing Service]

The selected database must handle high write throughput without sacrificing read
latency for dashboards. It must support horizontal scaling as device count grows.

However, under write-heavy load testing it saturated at 4,200 writes/sec on our
target hardware, which is well below the 12,000/sec peak requirement.

Apache Cassandra was evaluated for its linear horizontal scaling and
write-optimized LSM-tree storage engine. It achieved 18,500 writes/sec in our
benchmark, exceeding the peak requirement by 54%. Read latency for point lookups
was 3.2ms at P99, which is acceptable for dashboard refresh at 5-second intervals.

We selected Apache Cassandra as the event store because it is the only candidate
that satisfies both the write throughput requirement (12,000/sec) and the read
latency requirement (P99 < 10ms) simultaneously.

All writes must use the LOCAL_QUORUM consistency level to prevent data loss during
a single datacenter outage. Read operations may use LOCAL_ONE for lower latency.
Cassandra cluster must have a minimum of 3 nodes per datacenter.

PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required).
```

**Compression metadata:**
```
Original tokens:  ~450
Reduced tokens:   ~210
Compression:      53.3%
Sentences selected: 11 of 26
```

---

## 📊 Side-by-Side Comparison

| | Raw Context | BM25 | Vector Search | **SCRE** |
|--|:-----------:|:----:|:-------------:|:--------:|
| **Tokens** | 450 | 98 | 75 | 210 |
| **Decision sentence** | ✅ | ✅ | ✅ | ✅ |
| **Write constraints (LOCAL_QUORUM)** | ✅ | ✅ | ✅ | ✅ |
| **Min 3 nodes constraint** | ✅ | ❌ | ❌ | ✅ |
| **Performance evidence (18,500/sec)** | ✅ | ❌ | ❌ | ✅ |
| **P99 latency justification (3.2ms)** | ✅ | ❌ | ❌ | ✅ |
| **PostgreSQL rejection rationale** | ✅ | ⚠️ partial | ⚠️ partial | ✅ |
| **Noise (Background, Goals, Non-Goals)** | ❌ lots | ✅ none | ✅ none | ✅ none |
| **Implementation workflow** | ✅ (unwanted) | ❌ | ❌ | ✅ available |
| **Reasoning chain intact** | ✅ | ❌ | ❌ | ✅ |

---

## 🧠 Key Insight

**BM25 and Vector Search retrieve sentences. SCRE retrieves an argument.**

The critical difference for this query:

> The sentence *"It achieved 18,500 writes/sec in our benchmark, exceeding the peak requirement by 54%"* scores **low** on both BM25 (no query keywords) and Vector Search (subject is the pronoun "it", not "Cassandra"). It gets completely dropped by both baselines.

> SCRE retrieves it via the **reasoning graph** — it is causally linked to the decision sentence (sentence 15 has a `because` causal marker that points back to the outcome sentences 11, 12). The 2-hop traversal picks it up even though it wouldn't score well on its own.

This is exactly what the **Reasoning Graph Recall** metric measures:

| Strategy | Reasoning Graph Recall |
|----------|----------------------:|
| BM25 | 0.04% |
| Vector Search | 0.02% |
| **SCRE** | **12.45%** |

---

## 🔗 Related

- [`README.md`](README.md) — Project overview and quick start
- [`SCRE_DEEPDIVE.html`](SCRE_DEEPDIVE.html) — Full technical architecture and scoring formula
- [`tests/unified_benchmark.py`](tests/unified_benchmark.py) — Benchmark framework (75 docs, 100 Q&As)
- [`tests/benchmark_results.json`](tests/benchmark_results.json) — Raw results
