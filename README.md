# 🌿 GRACE: Graph-aware Reasoning And Context Engine

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-Research--Grade-purple)](README.md)
[![Benchmark](https://img.shields.io/badge/SPS-84.36%25-brightgreen)](tests/benchmark_results.json)
[![Docs](https://img.shields.io/badge/docs-GRACE--Deep--Dive-blue)](SCRE_DEEPDIVE.html)

> **ℹ️ NAMING:** This project was previously called **SCRE** (Sentence-level Context Reduction Engine). The name **GRACE** better reflects what actually differentiates it — graph-aware reasoning-chain preservation, not just sentence selection (see the benchmark below). This is a documentation-level rename only: the Python package, module names, and the `SCRE` class are unchanged for backward compatibility — every code example below still reads `from scre.query_aware_reducer import SCRE`.

> **ℹ️ STATUS:** GRACE is a **research-grade, benchmark-validated** library under active development (75-doc evaluation, 100 Q&A pairs, SPS 84.36). The core API (`reduce`) is stable. The extraction strategy chain and scoring weights are subject to change. **Not recommended for mission-critical production workloads** without independent evaluation on your target domain.

---

## 📖 Overview

**GRACE** (Graph-aware Reasoning And Context Engine, package name `scre`) is a query-aware context compression library built for Retrieval-Augmented Generation (RAG) pipelines and LLM serving systems. It is **not** a traditional vector search or BM25 retriever — it reasons about *semantic structure* and compresses documents while preserving the logical, causal, and procedural relationships that matter to your query.

### The Problem

Modern LLMs have massive context windows, but stuffing them with raw documents is counterproductive:

- 📈 **Increased Latency** — Longer context = slower generation
- 💸 **Higher API Costs** — Every token costs money
- 🤔 **Lost-in-the-Middle** — LLMs lose focus when surrounded by noise
- 🔗 **Broken Reasoning Chains** — Naive retrieval breaks causal links

GRACE sits between your retrieval layer and your LLM, acting as an intelligent semantic compression filter.

> 💡 **See it in action:** [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) walks through the same document and query across all four strategies — Raw Context, BM25, Vector Search, and GRACE — showing exactly what each retrieves and why.

---

## 🏗️ Architecture

```mermaid
graph TB
    A[Raw Document Input]
    B[Text Segmentation & Parsing]
    C[Semantic Unit Extraction]
    D[Knowledge Graph Construction]
    E[Reasoning Graph Construction]
    F[Dense Embeddings - all-MiniLM-L6-v2]
    G[User Query]
    H[Query-Aware Hybrid Scoring Engine]
    I[Top-K Unit Selection]
    J[Reasoning Chain Expansion]
    K[Adjacent Context Expansion]
    L[De-duplication & Ordering]
    M[Final Reduced Context]
    N[LLM Answer Generation]

    A --> B --> C
    C --> D
    C --> E
    C --> F
    G --> H
    D --> H
    E --> H
    F --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M --> N
```

---

## ⚙️ Pipeline Stages

GRACE executes in **six distinct stages** on every `reduce()` call:

### 1️⃣ Ingestion & Intelligent Parsing
- Raw text is split into semantic paragraphs and individual sentences
- Preserves complex structures: markdown lists, key-value headers, code blocks
- Detects and consolidates multi-line `Label:\nValue` patterns to prevent fragmentation
- Maintains narrative context across sentence boundaries

### 2️⃣ Semantic Unit Classification
Each sentence is classified into one of the following semantic types using a **two-stage strategy pipeline**:

| Type | Description | Example Signal |
|------|-------------|----------------|
| `decision` | Critical choices or selections | "chose", "decided", "selected" |
| `constraint` | Requirements or limitations | "must", "cannot", negation dependency (spaCy `neg`) |
| `reason` | Causal or justification statements | "because" |
| `implementation` | Concrete implementation detail | subject/object/relation triple present |
| `goal` | Purpose or mission statements | "goal", "protect", "preserve" |
| `workflow` | Ordered process steps | Markdown list following a header |
| `task` | Action items or TODOs | "implement", "create", "benchmark" |
| `risk` | Downsides or trade-offs of a choice | "risk", "downside", "trade-off" |
| `outcome` | Results or consequences | "resulted", "caused", "benefits", "consequence" |
| `alternative` | Options considered and rejected | "alternative", "rejected in favor of" |
| `comparison` | Trade-off or contrast | "versus", "faster than", "better" |
| `question` | Open/unresolved questions | Ends in "?", "open question" |
| `fact` | Definitions or factual statements | Default fallback |

**Extraction Strategy Chain:**
1. Regex-based SDLC tag matcher (`[DEC-01]`, `[CON-02]`, etc.)
2. Markdown key-value extractor (`Requirement: ...`, `Decision: ...`)
3. spaCy NLP dependency parser (POS tagging, named entity recognition, subtrees)

### 3️⃣ Dual Graph Construction

**Knowledge Graph** — Entity adjacency list:
- Connects subjects and objects across semantic units
- Powers connectivity scoring: units with high entity co-occurrence are ranked higher

**Reasoning Graph** — Causal chain graph:
- Links `decision → constraint`, `reason → decision`, `task → goal` etc.
- Detects lexical causal markers: `because`, `due to`, `depends on`, `satisfies`, `violates`
- Enables **multi-hop traversal** during retrieval

### 4️⃣ Query-Aware Hybrid Scoring
Each semantic unit is scored against the incoming query using a **composite formula**:

```
score = (type_weight × 2.0)
      + lexical_similarity   # IDF-weighted term overlap, non-linear multi-term boost
      + phrase_bonus         # +5.0 for exact phrase match
      + dense_bonus          # cosine similarity × [2–12] depending on confidence tier
      + entity_match         # +3.0 per named entity match
      + noun_match           # +2.0 per noun chunk match
      + relationship_dist    # graph connectivity bonus (capped at 1.0)
      + intent_bonus         # +3.0 when unit type aligns with query intent
      + hard_id_boost        # +15.0 per exact alphanumeric identifier match
```

Dense embeddings use `all-MiniLM-L6-v2` (fast inference, high quality). The evaluation suite uses `all-mpnet-base-v2` for stricter semantic similarity verification at `threshold ≥ 0.80`.

### 5️⃣ Context Expansion
After top-K selection, GRACE expands the result set:

- **Adjacent Context Expansion** (`context_window`) — Includes ±N neighbouring sentences to resolve pronouns and preserve local coherence
- **Reasoning Chain Expansion** — Performs 2-hop traversal on the reasoning graph to pull in causes and consequences of selected nodes

### 6️⃣ Final Assembly
- De-duplicates by Jaccard similarity (threshold 0.85)
- Re-orders by original sentence index to maintain narrative flow
- Injects context headers (`[Section Title]`) to prevent context detachment
- Returns clean context block + compression metrics

---

## 📊 Benchmark Results

### Evaluation Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | 75 documents (25 Kubernetes KEPs + 50 IETF RFCs) |
| **Q&A Pairs** | 100 (2 per KEP, 1 per RFC) |
| **Embedding Model** | `all-mpnet-base-v2` |
| **Similarity Threshold** | ≥ 0.80 cosine similarity |
| **Benchmark Script** | `tests/unified_benchmark.py` |
| **Document Selection** | Only documents with Summary + Motivation + Design + Alternatives + Constraints + Reasoning sections |

### What Each Metric Means

| Metric | One-liner |
|--------|-----------|
| **SPS Score** | *Composite score (0–100) combining all metrics below using weighted priorities. The headline number for comparing strategies.* |
| **Constraint Recall** | *Did the retrieved context preserve the rules and requirements? ("must", "cannot", "required" sentences)* |
| **Decision Traceability** | *Did the retrieved context preserve the choices made and why they were made?* |
| **Workflow Integrity** | *Did the retrieved context preserve process steps and procedural logic?* |
| **Reasoning Recall** | *Did the retrieved context preserve the motivation and justification behind decisions?* |
| **Reasoning Graph Recall** | *Did the retrieved context preserve causal chains — i.e. are cause and effect sentences both present together?* |
| **Dependency Recall** | *Did the retrieved context preserve cross-document references and citations?* |
| **Compression Ratio** | *What percentage of the original document was removed? Higher = fewer tokens sent to the LLM.* |
| **Avg Context Tokens** | *Average token count of the retrieved context across all 100 test queries.* |
| **Latency (ms)** | *Time to perform retrieval per query (excluding model loading).* |
| **SER** | *Semantic Efficiency Ratio — SPS score per token. How much semantic value per token spent.* |

### Strategy Comparison

> **⚠️ Sample run (20 Q&As / 19 docs), not the full 75-doc/100-Q&A suite** — see [BENCHMARK_REPORT_sample.md](BENCHMARK_REPORT_sample.md) for the raw report. Numbers below also add a **LangChain-style LLM-extraction baseline** (`LLMChainExtractor` — real generative inference via a local Ollama model over the raw document, the off-the-shelf default most RAG pipelines reach for). The previous full-suite numbers predate the taxonomy fixes that added `risk`/`alternative` categories and split the `reason`/`outcome` classification, so they're retired here rather than left stale; re-run `tests/unified_benchmark.py` for fresh full-suite figures.

| Metric | Raw Context | BM25 | Vector Search | LangChain LLMExtract | **GRACE** |
|--------|:-----------:|:----:|:-------------:|:---------------------:|:--------:|
| **SPS Score** | 98.42 | 76.17 | 78.25 | 84.00 | **83.17** |
| **Constraint Recall** | 100.0% | 86.7% | 88.3% | 93.3% | **90.0%** |
| **Decision Traceability** | 96.7% | 60.0% | 62.5% | 71.7% | **70.8%** |
| **Workflow Integrity** | 98.3% | 80.0% | 81.7% | 83.3% | **85.0%** |
| **Reasoning Recall** | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Reasoning Graph Recall** | 100.0% | 0.0% | 1.0% | 6.7% | **11.6%** |
| **Dependency Recall** | 100.0% | 0.0% | 0.08% | 11.6% | **29.1%** |
| **Compression Ratio** | 0.0% | 96.6% | 96.7% | 85.1% | **83.8%** |
| **Avg Context Tokens** | 4,953 | 90 | 87 | 439 | **499** |
| **Latency (ms)** | 0.001 | 0.49 | 65.7 | 87,450 | **2,347** |
| **SER (Score/Token)** | 215 | 7,617 | 7,825 | 2,074 | **1,805** |

### Key Findings

- **The LLM-extraction baseline scores marginally higher SPS (84.00 vs 83.17)** than GRACE, but only because it has an actual language model reading the full document — at **37x the latency** (87.5s vs 2.3s per query) and real generative-inference cost per call, versus GRACE's zero-LLM structural pipeline.
- **GRACE preserves structural relationships the LLM baseline misses**: 1.7x better Reasoning Graph Recall (11.6% vs 6.7%) and 2.5x better Dependency Recall (29.1% vs 11.6%) — an LLM asked to "extract relevant sentences" optimizes for topical relevance, not for keeping a decision linked to its reason or a workflow step linked to its predecessor.
- **BM25 and Vector Search destroy reasoning graph structure** — Reasoning Graph Recall drops near 0% because they return disconnected sentences without traversing causal chains.
- **Raw Context scores 98.4 SPS** but at 4,953 avg tokens, it provides no compression benefit.

### SPS Formula

```
SPS = (reasoning_recall × 0.30)
    + (constraint_recall × 0.25)
    + (decision_traceability × 0.20)
    + (workflow_integrity × 0.15)
    + (general_retention × 0.10)
    × 100
```

**Why these weights?** The weights are deliberate design choices reflecting what matters most in technical document RAG — they are not mathematically optimised. Reasoning (0.30) is weighted highest because a context that drops *why* a decision was made is misleading to an LLM. Constraints (0.25) are next because missing hard rules (`must`/`cannot`) can produce incorrect outputs. General factual retention (0.10) is lowest because that is what BM25 and Vector Search already do — it is the least differentiating capability. All weights sum to 1.0 and can be tuned per domain (e.g. a compliance use-case could raise constraints to 0.40).

---

## 📦 Installation

### From Source (Development)

```bash
git clone https://github.com/aparajithguha/Sentence_Level_Context_Reduction_Engine.git
cd Sentence_Level_Context_Reduction_Engine

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install in editable mode with all dependencies
pip install -e .[all]

# Download required spaCy NLP model
python -m spacy download en_core_web_sm
```

### Dependencies

| Component | Package | Purpose |
|-----------|---------|---------|
| **Core** | `spacy≥3.0.0` | NLP, entity recognition, POS tagging |
| **ML** | `sentence-transformers` | Dense semantic embeddings |
| | `torch` | Deep learning backend |
| | `tiktoken` | Accurate token counting (cl100k_base) |
| **Dashboard** | `streamlit` | Interactive evaluation UI |
| | `plotly` | Radar charts and network graphs |
| | `networkx` | Graph layout and visualization |
| **Eval** | `rank_bm25` | BM25 baseline comparisons |

---

## ⚡ Quick Start

### Basic Usage

```python
from scre.query_aware_reducer import SCRE

engine = SCRE()

document_text = """
The database chosen was PostgreSQL. It was selected because of its ACID compliance.
All writes must use transactions. Reads can use connection pooling.
The team decided to use read replicas for analytics workloads.
"""

result = engine.reduce(
    text=document_text,
    query="Why did we select PostgreSQL?",
    max_sentences=3,
    context_window=1
)

print(result["context"])
print(f"Compressed: {result['metadata']['reduction_ratio']:.1%}")
print(f"Tokens: {result['metadata']['reduced_estimated_tokens']}")
```

### Stateless by design

`reduce()` is the only entry point, and it is a single, self-contained,
in-memory computation over one document/query pair — nothing is written to
disk or retained after the call returns. GRACE is built to sit inline on a
live prompt path: each call is one prompt, and the reduced result is handed
to the LLM immediately. There is no document cache and no `document_id` —
calling `reduce()` again on the same text simply recomputes it.

The spaCy pipeline and sentence-transformer model *are* shared, process-wide
singletons (`scre.models.get_nlp()` / `get_embedder()`) — loading them is
expensive, so every `SCRE()` instance in a process reuses the same already-
loaded models rather than reloading them per call:

```python
from scre.query_aware_reducer import SCRE

engine = SCRE()
r1 = engine.reduce(document_text, "Why PostgreSQL?", max_sentences=4)
r2 = engine.reduce(document_text, "What are the constraints?", max_sentences=4)
```

### Introspection: `analyze()`

For visualizing or auditing the extraction/graph stages without scoring
against a query (e.g. the dashboard's reasoning-graph explorer), use
`analyze()`:

```python
result = engine.analyze(document_text)
result["units"]            # [{"sentence_index", "text", "unit_type", "render_text"}, ...]
result["reasoning_edges"]  # [(source_idx, target_idx), ...]
```

---

## 🖥️ Interactive Dashboard

GRACE ships with a research-grade Streamlit dashboard for exploration and evaluation:

```bash
streamlit run dashboard.py
```

**Dashboard Tabs:**
1. **📊 Performance Dashboard** — Full benchmark results table + radar chart comparison
2. **🔌 Graph Explorer** — Interactive network graph of reasoning edges per document
3. **🛝 Playground & Comparison** — Side-by-side BM25 vs Vector vs GRACE retrieval for any document/query pair

---

## 🎬 Showcase Demo (Web UI)

A lightweight Flask app for demonstrating *how* GRACE reduces a single prompt/document — paste content, see the reduced output, and inspect exactly which sentences were kept or dropped and why. It also detects and displays whether the input is a structured/system prompt or an unstructured (paragraph/image) prompt, since GRACE's reduction strategy differs for each.

```bash
python -m showcase.server
# open http://127.0.0.1:5050
```

`showcase/` is a pure consumer of `scre`'s public API (`SCRE` plus its documented submodule functions) — it never modifies the library itself, the same way any downstream application would use GRACE (package name `scre`) as a dependency.

---

## 🧪 Running Benchmarks

```bash
# Full evaluation suite (75 docs, 100 Q&As)
python tests/unified_benchmark.py

# View results
cat tests/benchmark_results.json

# Launch dashboard
streamlit run dashboard.py
```

---

## 🗂️ Project Structure

```
SCRE/
├── scre/                           # The library. Stable public API: scre.query_aware_reducer.SCRE
│   ├── query_aware_reducer.py      # SCRE facade class: orchestrates the pipeline below
│   ├── units.py                    # SemanticUnit/WorkflowUnit data model + CATEGORY_WEIGHTS taxonomy
│   ├── extraction.py               # Segmentation + regex/NLP extraction strategy chain
│   ├── graph.py                    # Knowledge graph + reasoning graph construction
│   ├── scoring.py                  # Query-aware hybrid scoring
│   ├── selection.py                # Top-K selection, adjacent + reasoning-chain expansion
│   ├── assembly.py                 # De-duplication, ordering, final context assembly
│   ├── config.py                   # ScoringConfig thresholds
│   ├── models.py                   # Process-wide singleton spaCy/embedder loaders
│   ├── scre_pipeline.py            # End-to-end pipeline (reduce → answer)
│   └── scre_answer_engine.py       # LLM integration (Ollama) for answering/judging, not reduction
│
├── showcase/                       # Separate demo web app -- a *consumer* of scre's public API only,
│   │                               # never modifies the library. Not part of the installable package.
│   ├── reducer_adapter.py          # ExplainedReducer: reduce() with per-unit scoring + prompt-type detection
│   ├── server.py                   # Flask app serving the demo UI + JSON API
│   └── static/index.html           # Single-page UI: paste content, see the reduction + why
│
├── tests/
│   ├── unified_benchmark.py        # Research-grade evaluation suite (75 docs, 100 Q&As)
│   ├── benchmark_results.json      # Latest benchmark output
│   ├── benchmark_results.csv       # CSV export for analysis
│   └── benchmark_history.json      # Run history
│
├── data/                           # Benchmark corpora & agent prompt templates
├── dashboard.py                    # Streamlit interactive dashboard
├── WORKFLOW_EXAMPLE.md             # Step-by-step retrieval comparison walkthrough
├── SCRE_DEEPDIVE.html              # Full technical deep-dive document
└── pyproject.toml                  # Project configuration
```

---

## 🔧 Configuration

### GRACE (`SCRE`) Constructor

```python
engine = SCRE(
    model="en_core_web_sm",   # spaCy model name (fetched from the shared singleton cache)
    nlp=None,                 # Optional: an already-loaded spaCy pipeline to use instead
    embedder=None,            # Optional: an already-loaded SentenceTransformer to use instead
    extractors=[...]          # Optional: custom extraction strategy chain
)
```

### Reduce Parameters

```python
result = engine.reduce(
    text="...",
    query="...",
    max_sentences=6,       # Max semantic units to select
    context_window=1,      # Adjacent sentence expansion radius
    min_tokens=250,        # Minimum token floor (prevents under-retrieval; capped
                            # at 2x max_sentences and at the tokens actually available)
    max_tokens=None        # Optional hard token ceiling
)
```

### Return Schema

```python
{
    "context": "...",          # The compressed context string
    "metadata": {
        "original_sentences": 660,
        "reduced_sentences": 15,
        "original_chars": 45769,
        "reduced_chars": 3641,
        "original_estimated_tokens": 10325,
        "reduced_estimated_tokens": 877,
        "reduction_ratio": 0.9204
    }
}
```

---

## 🎯 Use Cases

### RAG Pipeline Compression
```python
retrieved_docs = vector_db.search(query, top_k=10)
full_context = "\n".join(retrieved_docs)

compressed = engine.reduce(full_context, query, max_sentences=5)
answer = llm.chat(query, context=compressed["context"])
```

### Cost-Aware Production Serving
```python
result = engine.reduce(text=raw_doc, query=user_query, max_sentences=3)
token_savings = (result["metadata"]["original_estimated_tokens"]
                 - result["metadata"]["reduced_estimated_tokens"])
print(f"Saved {token_savings} tokens (~${token_savings * 0.000003:.4f} at GPT-4 pricing)")
```

---

## 🚀 Roadmap

- [x] Semantic unit extraction with type classification
- [x] Dual-graph construction (Knowledge + Reasoning)
- [x] Dense + sparse hybrid scoring
- [x] Reasoning chain multi-hop expansion
- [x] Stateless, in-memory reduction with shared model singletons
- [x] Research-grade benchmark suite (75 docs, 100 Q&As)
- [x] Interactive Streamlit dashboard with graph explorer
- [ ] LangChain / LlamaIndex retriever integration
- [ ] Multi-language support (multilingual MiniLM)
- [ ] Streaming context expansion
- [ ] Fine-tuned domain-specific extraction models
- [ ] Production monitoring & observability hooks

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — project overview and quick start |
| [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) | Step-by-step retrieval comparison (Raw vs BM25 vs Vector vs GRACE) |
| [SCRE_DEEPDIVE.html](SCRE_DEEPDIVE.html) | Full technical architecture, scoring formula, and benchmark analysis |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | Auto-generated benchmark narrative report |
| [BENCHMARK_REPORT_sample.md](BENCHMARK_REPORT_sample.md) | Fast-iteration comparison (20-question / 19-doc sample, not the full 75-doc/100-Q&A suite) adding a LangChain-style LLM-extraction baseline (`LLMChainExtractor`) alongside Raw/BM25/Vector/GRACE |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with [spaCy](https://spacy.io/), [Sentence-Transformers](https://www.sbert.net/), [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), and [NetworkX](https://networkx.org/).
