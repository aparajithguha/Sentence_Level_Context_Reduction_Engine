


## 🏆 SCRE Publishable Evaluation Suite (19 Docs, 20 Q&As)
*Defensible research-grade evaluation comparing raw proposal context, BM25 keyword matching, dense Vector Search, LLM-extraction (LangChain-style), and the structural SCRE engine on semantic recall, graph paths, and step dependencies.*
**Run Date:** 2026-09-14 20:52:40

| Strategy | SPS Score (0-100) | Constraint Recall | Decision Traceability | Workflow Integrity | Reasoning Recall | Reasoning Graph Recall | Dependency Recall | SER (Efficiency) | Compression | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Raw Context** | 98.42 | 100.00% | 96.67% | 98.33% | 100.00% | 100.00% | 100.00% | 215.27 | 0.00% | 0.0ms |
| **BM25** | 76.17 | 86.67% | 60.00% | 80.00% | 100.00% | 0.00% | 0.00% | 7617.00 | 96.55% | 0.5ms |
| **Vector Search** | 78.25 | 88.33% | 62.50% | 81.67% | 100.00% | 1.00% | 0.08% | 7825.00 | 96.68% | 65.7ms |
| **SCRE** | 83.17 | 90.00% | 70.83% | 85.00% | 100.00% | 11.59% | 29.14% | 1804.90 | 83.83% | 2347.4ms |
| **LangChain LLMExtract** | 84.00 | 93.33% | 71.67% | 83.33% | 100.00% | 6.71% | 11.56% | 2074.07 | 85.07% | 87450.4ms |
