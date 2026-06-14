# ✂️ SCRE: Sentence-level Context Reduction Engine

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **⚠️ DISCLAIMER:** SCRE is currently an **experimental package**. It is in active development, and internal APIs (especially semantic extraction strategies) may change rapidly. It is not yet recommended for mission-critical production workloads without thorough evaluation.

## 📖 What is this Library?

**SCRE** (Sentence-level Context Reduction Engine) is a query-aware context compression library designed for Retrieval-Augmented Generation (RAG) pipelines.

Modern LLMs have massive context windows, but stuffing them with unrefined documents is an anti-pattern that increases latency, spikes API costs, and causes "lost-in-the-middle" hallucinations. SCRE sits between your vector database and your LLM. It intelligently reduces the retrieved context down to the *exact sentences* needed to answer a user's query, while simultaneously preserving critical reasoning chains, constraints, and narrative flow.

---

## ⚙️ How It Works

SCRE operates through a multi-stage pipeline that is executed every time you call `engine.reduce()`:

1.  **Ingestion & Parsing:** The raw text is ingested and intelligently segmented into sentences. It handles complex structures like markdown lists, headers, and key-value pairs to avoid breaking them apart.
2.  **Semantic Unit Extraction:** Each sentence is processed by a series of strategies (first regex for structured data, then NLP as a fallback) to classify it into a `SemanticUnit` (e.g., `decision`, `constraint`, `workflow`, `fact`).
3.  **Graph Construction:** Two graphs are built in memory: a **Knowledge Graph** connecting entities (subjects/objects) and a **Reasoning Graph** connecting causal or logical steps (e.g., a `decision` linked to its `reason`).
4.  **Query-Aware Scoring:** Each `SemanticUnit` is scored against the user's query using a hybrid algorithm that considers:
    *   **Dense Semantic Similarity** (Vector embeddings)
    *   **Sparse Lexical Similarity** (IDF-weighted term overlap)
    *   **Entity & Noun Chunk Matching**
    *   **Structural & Intent Bonuses** (e.g., boosting `workflow` units for "how-to" questions)
5.  **Context Expansion:** The top-scoring sentences are selected. This selection is then expanded to include:
    *   **Adjacent Sentences** (`context_window`) to resolve pronouns and local context.
    *   **Reasoning Chains** by traversing the reasoning graph to pull in related causes or outcomes.
6.  **Final Assembly:** The final set of sentences is de-duplicated, ordered, and formatted into a clean context block, ready to be sent to the LLM.

---

## 📦 Installation

SCRE uses a modern `pyproject.toml` configuration. You can install it directly from source in editable mode with all required dependencies:

### Method 1: From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-username/scre.git # <-- Update with your repo URL
cd scre

# Install the package in editable mode with all dependencies
pip install -e .[all]

# Download the required spaCy NLP model
python -m spacy download en_core_web_sm
```

---

## ⚡ Quick Start

SCRE is designed to drop right into your existing Python applications.

```python
from scre.query_aware_reducer import SCRE

# 1. Initialize the Engine
engine = SCRE()

document_text = """
John owns Project Phoenix. Sarah manages Project Phoenix.
Mike tests Project Phoenix. David deploys Project Phoenix.
The database chosen was PostgreSQL. It was selected because of its ACID compliance.
"""

query = "Why did we select PostgreSQL?"

# 2. Reduce the Context
result = engine.reduce(
    text=document_text,
    query=query,
    max_sentences=2,
    context_window=1  # Pulls in adjacent sentences to preserve reasoning
)

print("--- Reduced Context ---")
print(result["context"])

print("\n--- Compression Metrics ---")
print(f"Reduction Ratio: {result['metadata']['reduction_ratio']:.2%}")
print(f"Reduced Tokens: {result['metadata']['reduced_estimated_tokens']}")
```

---

## 📊 Benchmarking Performance

SCRE includes a rigorous benchmarking framework (`benchmark_v2.py`) to prove its efficacy against traditional Raw Context, BM25, and Vector Search retrieval strategies.

The benchmark evaluates:
* **Exact Match & F1 Score** (Against expected ground-truth answers)
* **Semantic Similarity**
* **Answer Consistency** (Does the reduced context yield the *same* answer as full context?)
* **Compression Ratio & Cost Savings**

**Run the benchmark:**
```bash
python tests/benchmark_v2.py
```
*(Results are automatically tracked historically in `BENCHMARK_REPORT.md`)*

---

## 🗂️ Project Structure

```text
SCRE/
├── scre/                       # Core Library
│   ├── query_aware_reducer.py  # Graph-based sentence extraction & scoring
│   ├── scre_answer_engine.py   # RAG QA execution (Extractive & Ollama Support)
│   └── scre_pipeline.py        # End-to-end integration wrappers
├── tests/                      # Testing & Benchmarking
│   ├── benchmark_v2.py         # Advanced multi-strategy evaluation framework
│   ├── scre_eval.py            # Local evaluation harness (Meaning Retention Score)
│   └── qa_dataset.py           # Evaluation ground-truth datasets
├── data/                       # Benchmark Test Data
│   ├── sdlc_project_benchmark.txt
│   └── bench_answers.txt
└── pyproject.toml              # Modern Python packaging configuration
```

---

## 🤝 Current Direction

SCRE has evolved from a naive triple-extraction prototype to a fully mature **query-conditioned reduction** engine. Future roadmaps include:
* Native LangChain and LlamaIndex retriever abstractions.
* Distributed SQLite optimizations for massive batch processing.
* Deep integration with structured JSON/Markdown formats (Workflows, SDLC formats).

## Current direction

This repo no longer uses the older graph/triple compression prototype.
The working direction is query-conditioned reduction, because token savings only
matter if answer fidelity is preserved for a specific ask.
