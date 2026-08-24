# CHATBOT-ENG-GUILD: Pending TODOs & Backlog

## 1. Retrieval Evaluation & Benchmark Harness (Deferred T4)

* **Task:** Build a robust, accurate evaluation suite for the RAG retrieval engine.
* **Reason Deferred:** The initial prototype used loose keyword substring matching (`any(kw in text)`), leading to unrepresentative 100% recall metrics.
* **Requirements for Implementation:**
  1. **Strict Ground-Truth Grounding:** Map evaluation queries to exact target Document IDs / Chunk IDs or `expected_doc_type` metadata.
  2. **Multi-Keyword & Semantic Scoring:** Replace simple string matching with strict semantic alignment or LLM-as-a-judge (RAGAS / G-Eval framework).
  3. **Expanded Dataset:** Expand the dialect query dataset from 10 to 50+ diverse JEA queries covering all service areas.
  4. **Metrics to Measure:** Recall@5, Recall@15, and MRR (Mean Reciprocal Rank) after Phase 2 context and retriever optimization.

---

## 2. Unit & Integration Test Suite Enhancements

### A. Vector Embedding Retrieval Testing (`tests/unit/test_retrieval.py`)
* **Task:** Expand retrieval testing beyond pure rank-math to include live ChromaDB vector search and BM25 sparse index query validation.
* **Objective:** Ensure embedding distance calculations, similarity scoring, and dense+sparse candidate retrieval work accurately against real document vectors.

### B. Dynamic Corrective RAG & Router Testing (`tests/unit/test_corrective_rag.py`)
* **Task:** Enhance Corrective RAG and Router tests once dynamic routing logic (Phase 2) is connected.
* **Objective:** Replace deterministic route assertions with dynamic branching checks (verifying `LOW` confidence triggers query expansion / fallback routing while `HIGH` confidence routes to generation).
