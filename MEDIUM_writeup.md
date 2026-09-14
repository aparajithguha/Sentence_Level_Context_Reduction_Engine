# I Built a Context Compression Engine That Understands *Why* — Not Just *What*

## Introducing SCRE: Sentence-level Context Reduction Engine — benchmark results and the exact problem that sparked it

---

> *"The LLM had all the information. It just couldn't find the reason."*

That sentence summarises about six months of frustration building production RAG pipelines.

---

## The Quiet Failure Mode Nobody Talks About

Everyone building with LLMs knows the standard RAG stack:

1. Chunk your documents
2. Embed them
3. Vector search for the top-K chunks
4. Stuff them into the prompt
5. Hope the model figures it out

And for a lot of queries, it works. Ask "What is the capital of France?" — great. Ask "What did the PostgreSQL ADR say?" — fine.

But then you get the hard questions. The ones that actually matter in enterprise software:

- *"Why did we choose Cassandra over PostgreSQL for the event store?"*
- *"What are the operational constraints on our Kubernetes cluster that drove this decision?"*
- *"Given the constraints in RFC-8888, what's the reasoning chain that led to the authentication design?"*

These are **causal reasoning questions**. And this is where every standard retrieval strategy silently, invisibly fails — and hands a hallucinating LLM a context full of disconnected facts.

---

## The Problem, Made Concrete

Let me show you exactly what I mean with a real example.

Consider a 450-token Architecture Decision Record (ADR) describing why a team chose Apache Cassandra as their event store. You ask:

> *"Why was Apache Cassandra selected, and what are the write constraints?"*

This query has two parts: a **decision trace** ("why was it chosen?") and a **constraint extraction** ("what are the rules?"). The answer spans four different sections of the document.

### What BM25 returns

BM25 tokenises the query and scores sentences by term frequency. It returns the top 6 sentences with "cassandra", "write", "selected", "constraints" in them. You get:

```
"Apache Cassandra was evaluated for its linear horizontal scaling..."
"We selected Apache Cassandra as the event store because..."
"All writes must use the LOCAL_QUORUM consistency level..."
"Redis Streams was fast (>50,000 writes/sec) but offers no durability..."
```

That looks reasonable on the surface. But look at what's missing:

❌ **"It achieved 18,500 writes/sec in our benchmark, exceeding peak by 54%."**

That sentence is the *entire justification* for the decision. It's the critical performance evidence. But BM25 scores it low — the subject is the pronoun "it", not "Cassandra". There are no query keywords in it. So the LLM never sees it.

### What Vector Search returns

Vector search does better. It understands semantic meaning. The decision sentence scores 0.91 cosine similarity. The constraint sentence scores 0.84. Great.

But that critical benchmark sentence — "It achieved 18,500 writes/sec" — still scores low. The embedding of "It achieved..." doesn't cluster near "Why was Cassandra selected?" because "it" has no referent in isolation. The model doesn't know that sentence is about Cassandra.

Vector search is *better* than BM25. But it still treats every sentence as an **isolated island**.

### What this does to your LLM

Your model receives a context with:
- The decision ✅
- The write constraints ✅
- Some noise from rejected alternatives ⚠️
- **Missing: the performance evidence that justified the decision** ❌

The LLM will confidently explain *that* Cassandra was chosen. But it can't explain *why with evidence* — because the evidence was never retrieved. This is the lost-in-the-reasoning-chain problem.

---

## The Insight That Changed Everything

Here's the core realisation that drove SCRE:

> **BM25 and Vector Search retrieve sentences. What we actually need is to retrieve an *argument*.**

An argument has structure. It has:
- A **claim** (the decision)
- **Evidence** (the benchmark numbers)
- **Constraints** (the operational rules that follow from the decision)
- **Alternatives rejected** (the contrast that makes the decision meaningful)

These are causally linked. And causality is invisible to both keyword matching and embedding cosine similarity.

---

## Introducing SCRE

**SCRE** (Sentence-level Context Reduction Engine) is a query-aware context compression library that sits between your retrieval layer and your LLM. Instead of treating documents as bags of sentences, it models them as **semantic graphs** — and traverses those graphs to answer your query.

It runs in six stages:

### Stage 1 — Intelligent Parsing

SCRE splits documents into semantic units that respect structure. Ordered lists become atomic `WorkflowUnit` objects. Multi-line `Label: Value` patterns are consolidated. Markdown headers are tracked to prevent context detachment.

### Stage 2 — Semantic Classification

Every sentence is classified into one of nine semantic types using a three-stage extraction strategy chain:

| Type | Signal |
|------|--------|
| `decision` | "chose", "selected", "decided" |
| `constraint` | "must", "cannot", "required" |
| `reason` | "because", "due to", "enables" |
| `outcome` | "achieved", "resulted", "caused" |
| `workflow` | Ordered markdown list after a header |
| `goal` | "goal", "protect", "preserve" |
| `comparison` | "versus", "faster than", "better" |
| `fact` | Default NLP fallback |
| `task` | "implement", "create", "benchmark" |

The pipeline tries regex SDLC tags first, then markdown key-value patterns, then falls back to a spaCy NLP dependency parser for POS tagging and named entity recognition.

### Stage 3 — Dual Graph Construction

This is the core architectural innovation.

**Knowledge Graph** — An entity adjacency list that connects subjects and objects across semantic units. Enables connectivity scoring: sentences that share key entities score higher together.

**Reasoning Graph** — A causal chain graph. It links `decision → constraint`, `reason → decision`, `task → goal`. It detects lexical causal markers: `because`, `due to`, `depends on`, `satisfies`, `violates`.

For the ADR example, the Reasoning Graph looks like this:

```
15 (decision: Cassandra selected)
  ← 11  (outcome: 18,500 writes/sec achieved)    # "because it satisfies"
  ← 12  (fact: P99 3.2ms latency)               # adjacent causal
  ← 5   (constraint: must handle throughput)     # decision satisfies constraint
  → 17  (constraint: LOCAL_QUORUM required)      # decision drives constraint
  → 19  (constraint: min 3 nodes)               # decision drives constraint
```

### Stage 4 — Hybrid Scoring

Each sentence is scored against the query using a composite formula:

```
score = (type_weight × 2.0)
      + lexical_similarity      # IDF-weighted term overlap
      + phrase_bonus            # +5.0 for exact phrase match
      + dense_bonus             # cosine similarity × [2–12] by confidence tier
      + entity_match            # +3.0 per named entity match
      + noun_match              # +2.0 per noun chunk match
      + relationship_dist       # graph connectivity bonus (capped at 1.0)
      + intent_bonus            # +3.0 when unit type aligns with query intent
      + hard_id_boost           # +15.0 per exact alphanumeric identifier match
```

The query intent is parsed to detect whether the user is asking for a decision trace, constraint extraction, workflow, or general fact retrieval — and weights are shifted accordingly.

### Stage 5 — Reasoning Chain Expansion

After selecting top-K sentences by score, SCRE performs **2-hop graph traversal** on the Reasoning Graph. Causes and consequences of selected nodes are pulled in.

For our example: the outcome sentence "It achieved 18,500 writes/sec" is causally linked to the decision sentence. Even though it scores low on hybrid scoring alone, it gets included via traversal.

This is exactly what BM25 and Vector Search cannot do.

### Stage 6 — Final Assembly

- De-duplicates by Jaccard similarity (threshold 0.85)
- Re-orders by original sentence index (maintains narrative flow)
- Injects `[Section Title]` headers to prevent context detachment
- Returns a clean context block + full compression metrics

---

## The Output That Proves It Works

For the Cassandra ADR, SCRE returns **~210 tokens** (53% compression) containing:

```
[ADR-007: Database Selection for Event Processing Service]

The selected database must handle high write throughput without
sacrificing read latency for dashboards.

However, under write-heavy load testing it saturated at 4,200 writes/sec
on our target hardware, which is well below the 12,000/sec peak requirement.

Apache Cassandra was evaluated for its linear horizontal scaling and
write-optimized LSM-tree storage engine. It achieved 18,500 writes/sec
in our benchmark, exceeding the peak requirement by 54%.
Read latency for point lookups was 3.2ms at P99.

We selected Apache Cassandra as the event store because it is the only
candidate that satisfies both the write throughput requirement (12,000/sec)
and the read latency requirement (P99 < 10ms) simultaneously.

All writes must use the LOCAL_QUORUM consistency level to prevent data
loss during a single datacenter outage. Read operations may use LOCAL_ONE.
Cassandra cluster must have a minimum of 3 nodes per datacenter.

PostgreSQL: Failed throughput benchmark (4,200/sec vs 12,000/sec required).
```

The context is now a **complete argument**: claim → evidence → constraints → rejected alternatives. The LLM sees not just *what* was decided, but *why*, with the numbers to back it up.

Compared to the baselines:

| | Raw Context | BM25 | Vector Search | **SCRE** |
|--|:-----------:|:----:|:-------------:|:--------:|
| Tokens | 450 | 98 | 75 | **210** |
| Decision sentence | ✅ | ✅ | ✅ | ✅ |
| Write constraints | ✅ | ✅ | ✅ | ✅ |
| Min 3-nodes constraint | ✅ | ❌ | ❌ | ✅ |
| Performance evidence (18,500/sec) | ✅ | ❌ | ❌ | ✅ |
| P99 latency justification | ✅ | ❌ | ❌ | ✅ |
| Reasoning chain intact | ✅ | ❌ | ❌ | ✅ |

---

## Benchmark Results: 75 Documents, 100 Q&A Pairs

I evaluated SCRE against Raw Context, BM25, and Vector Search on a dataset of 75 technical documents — 25 Kubernetes Enhancement Proposals (KEPs) and 50 IETF RFCs. 100 question-answer pairs were designed to test exactly the hard cases: constraint recall, decision traceability, reasoning chain preservation.

The headline metric is **SPS (Semantic Preservation Score)** — a composite score that weights:
- Reasoning Recall: 0.30 *(highest — dropping the "why" is the worst failure)*
- Constraint Recall: 0.25 *(missing hard rules produces wrong outputs)*
- Decision Traceability: 0.20
- Workflow Integrity: 0.15
- General Retention: 0.10 *(lowest — BM25 already does this)*

| Metric | Raw Context | BM25 | Vector Search | **SCRE** |
|--------|:-----------:|:----:|:-------------:|:--------:|
| **SPS Score** | 97.40 | 76.53 | 77.32 | **84.36** |
| **Constraint Recall** | 97.0% | 76.0% | 77.3% | **82.8%** |
| **Decision Traceability** | 96.7% | 77.2% | 77.7% | **85.8%** |
| **Reasoning Graph Recall** | 100.0% | **0.04%** | **0.02%** | **12.45%** |
| **Compression Ratio** | 0.0% | 96.0% | 96.1% | **80.5%** |
| **Avg Context Tokens** | 5,796 | 92 | 90 | **518** |

The number that jumps out: **Reasoning Graph Recall**. BM25 scores 0.04%. Vector Search scores 0.02%. SCRE scores 12.45%.

This metric asks: *are cause and effect sentences both present together in the retrieved context?* BM25 and Vector Search essentially destroy causal structure. SCRE is the only compressed retrieval method that starts to rebuild it.

Decision Traceability at **85.8%** also exceeds both baselines by ~8.6 percentage points — SCRE follows decision-reason-implementation chains, not just keyword co-occurrence.

---

## The Practical Value

Here's what this translates to in a production system:

**Token savings at GPT-4 scale:**
```python
result = engine.reduce(text=raw_doc, query=user_query, max_sentences=3)
token_savings = (
    result["metadata"]["original_estimated_tokens"]
    - result["metadata"]["reduced_estimated_tokens"]
)
print(f"Saved {token_savings} tokens (~${token_savings * 0.000003:.4f} at GPT-4 pricing)")
```

At 80.5% average compression, a pipeline processing 10,000 queries/day against 5,000-token documents goes from ~50M tokens/day to ~9.75M tokens/day — roughly an **80% cost reduction** while *improving* the semantic quality of context the LLM receives.

**Plugging into an existing RAG pipeline:**
```python
from scre.query_aware_reducer import SCRE

engine = SCRE()

# Drop-in replacement for your context assembly step
retrieved_docs = vector_db.search(query, top_k=10)
full_context = "\n".join(retrieved_docs)

compressed = engine.reduce(full_context, query, max_sentences=5)
answer = llm.chat(query, context=compressed["context"])
```

One line change. Reasoning-aware context from that point on.

---

## What SCRE Is — and Isn't

SCRE is **not** a replacement for your vector database. It's not a retriever. It doesn't compete with Pinecone, Weaviate, or Chroma.

SCRE is a **semantic compression filter** that sits after retrieval and before the LLM prompt. It takes whatever context you've assembled — from vector search, BM25, or both — and compresses it while preserving the causal structure that cheap retrieval methods destroy.

Think of it as the difference between handing an analyst a 10,000-word document and handing them a structured briefing with the decision, the evidence, and the constraints — in 1,200 words.

---

## Current State and Roadmap

SCRE is **research-grade and open source**. The core APIs (`ingest`, `retrieve`, `reduce`) are stable and benchmark-validated. The extraction strategy chain and scoring weights are under active development.

**What's working now:**
- ✅ Dual-graph construction (Knowledge + Reasoning)
- ✅ Dense + sparse hybrid scoring
- ✅ 2-hop reasoning chain expansion
- ✅ Persistent SQLite-backed memory store
- ✅ Research-grade benchmark suite (75 docs, 100 Q&As)
- ✅ Interactive Streamlit dashboard with graph explorer

**What's coming:**
- 🔲 LangChain and LlamaIndex retriever integration
- 🔲 Multi-language support (multilingual MiniLM)
- 🔲 Fine-tuned domain-specific extraction models
- 🔲 Production observability hooks

---

## The Dashboard

SCRE ships with a Streamlit dashboard for exploration:

```bash
git clone https://github.com/aparajithguha/Sentence_Level_Context_Reduction_Engine.git
cd Sentence_Level_Context_Reduction_Engine
pip install -e .[all]
python -m spacy download en_core_web_sm
streamlit run dashboard.py
```

Three tabs:
1. **Performance Dashboard** — Full benchmark results + radar chart comparison
2. **Graph Explorer** — Interactive network graph of reasoning edges per document
3. **Playground** — Side-by-side BM25 vs Vector vs SCRE for any document/query pair

---

## Closing Thought

RAG is not a solved problem. Retrieval accuracy has gotten better. Embedding models have gotten better. But the compression and context-assembly step — the translation from "a pile of retrieved chunks" to "the exact argument the LLM needs" — has largely been left to naive chunking and top-K selection.

SCRE is my attempt to fix that specific problem. Not by making retrieval smarter, but by making the compression step *structurally aware* — so the reasoning chain from evidence to decision to constraint isn't accidentally destroyed before the model ever sees it.

The benchmark numbers suggest it's working. The hard part now is making it fast enough and general enough for production at scale.

If you're building RAG pipelines, dealing with technical document Q&A, or just frustrated that your LLM gives you confident answers with no supporting evidence — I'd love your feedback.

**GitHub:** [Sentence_Level_Context_Reduction_Engine](https://github.com/aparajithguha/Sentence_Level_Context_Reduction_Engine)

---

*Built with spaCy, Sentence-Transformers, Streamlit, Plotly, and NetworkX. MIT License.*

---

### Tags
`#LLM` `#RAG` `#NLP` `#MachineLearning` `#Python` `#OpenSource` `#ContextCompression` `#KnowledgeGraph` `#ArtificialIntelligence` `#SoftwareEngineering`
