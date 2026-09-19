


## 🏆 SCRE Publishable Evaluation Suite (19 Docs, 20 Q&As)
*Defensible research-grade evaluation comparing raw proposal context, BM25 keyword matching, dense Vector Search, LLM-extraction (LangChain-style), and the structural SCRE engine on semantic recall, graph paths, and step dependencies.*
**Run Date:** 2026-09-20 01:39:13

| Strategy | SPS Score (0-100) | Constraint Recall | Decision Traceability | Workflow Integrity | Reasoning Recall | Reasoning Graph Recall | Dependency Recall | SER (Efficiency) | Compression | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Raw Context** | 98.42 | 100.00% | 96.67% | 98.33% | 100.00% | 100.00% | 100.00% | 215.27 | 0.00% | 0.0ms |
| **BM25** | 76.17 | 86.67% | 60.00% | 80.00% | 100.00% | 0.00% | 0.00% | 7617.00 | 96.55% | 0.6ms |
| **Vector Search** | 78.25 | 88.33% | 62.50% | 81.67% | 100.00% | 0.88% | 0.00% | 7825.00 | 96.68% | 70.9ms |
| **SCRE** | 84.17 | 90.00% | 73.33% | 86.67% | 100.00% | 14.27% | 50.00% | 2078.27 | 84.47% | 2629.7ms |
| **LangChain LLMExtract †** | 84.00 | 93.33% | 71.67% | 83.33% | 100.00% | 6.71% | 11.56% | 2074.07 | 85.07% | 87450.4ms |

† Not re-run: taken from the saved run of 2026-09-14 20:52:40 on the same 20 questions. It reads the raw documents and does not depend on SCRE's code.
