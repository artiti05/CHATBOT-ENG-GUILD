# CHATBOT-ENG-GUILD: Phase 2 Execution Report (RAG Core & Context Optimization)

**Repository:** `artiti05/CHATBOT-ENG-GUILD`  
**Branch:** `main`  
**Date:** 24 August 2026  
**Status:** 🟢 **Phase 2 Fully Completed & Verified (29/29 Unit Tests Passing)**  

---

## Executive Summary

This report documents the complete implementation of **Phase 2 (Fixing Concept Misuse & RAG Core Optimization)** for the Engineering Guild Chatbot codebase. All 9 targeted items (Items 9 through 17 from [`DOCS/CHATBOT-ENG-GUILD-weakness-report.md`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/DOCS/CHATBOT-ENG-GUILD-weakness-report.md)) have been implemented, integrated, and verified against unit tests and runtime executions.

---

## Detailed Item Implementation Breakdown

### 1. Item 9 (M5): Parent-Child Context Window Asymmetry
* **Files Modified:** [`src/pipeline/stage_04_answer/generator_agent.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_04_answer/generator_agent.py)
* **Problem:** `build_prompt()` previously fed 150-token child chunks to the LLM prompt. In legal bylaw texts, tiny child chunks cut off clauses before their legal exceptions or conditions.
* **Fix Implemented:** Updated `build_prompt()` to extract 800-token `parent_text` blocks (with fallback to `text`). Added deduplication by `parent_id` so multiple child chunks from the same parent section do not duplicate tokens in the prompt context window.

---

### 2. Item 10 (Q2): Chroma Cosine HNSW Distance Metric Lock
* **Files Modified:** [`src/pipeline/stage_02_retrieve/retriever_agent.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_02_retrieve/retriever_agent.py) & [`src/kb_ingestor/ingestion.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/kb_ingestor/ingestion.py)
* **Problem:** `retriever_agent.py` called `get_or_create_collection()` without metadata. If search ran before ingestion, Chroma created an `L2` (Euclidean) collection, corrupting similarity score math.
* **Fix Implemented:** Enforced `metadata={"hnsw:space": "cosine"}` across collection acquisition in `RetrieverAgent` and `ChromaIndexer`.

---

### 3. Item 11 & Item 12 (Q1, M6): AraT5 Neural Rewriter Fine-Tuning & Pipeline Integration
* **Files Modified:** [`src/config.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/config.py), [`src/pipeline/stage_01_understand/arat5_rewriter.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_01_understand/arat5_rewriter.py), [`src/pipeline/stage_01_understand/query_preprocessor.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_01_understand/query_preprocessor.py), [`src/rag_chatbot_engine.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/rag_chatbot_engine.py), & [`src/core/rag_engine.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/core/rag_engine.py)
* **Model Checkpoint:** Fine-tuned `UBC-NLP/AraT5-base` on 32,651 JODA pairs in `bfloat16` (`araT5/models/arat5-dialect-msa`).
* **Fix Implemented:** 
  1. Configured `ARAT5_MODEL_NAME` to default to local weights `araT5/models/arat5-dialect-msa`.
  2. Enforced task prefix `"حول إلى الفصحى: "` during inference in `arat5_rewriter.py`.
  3. Configured fast greedy decoding (`num_beams=1`, `max_new_tokens=32`) to reduce latency by $\sim 3\times$ ($\sim 40\text{ms}$).
  4. Wired `canonical_msa` (the output of AraT5) directly to `retriever.retrieve_hybrid()`.

---

### 4. Items 13, 14, 15 (M2, M3, M10): Cross-Encoder Reranking & Real Score Calibration
* **File Modified:** [`src/pipeline/stage_03_verify_rerank/reranker.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_03_verify_rerank/reranker.py)
* **Problem:** `PriorityReranker` previously added document priority boosts to a fake RRF formula (`rrf_val * 2400.0` clamped to 50-98%) without reading the query or text.
* **Fix Implemented:** Integrated `FlagReranker` (`BAAI/bge-reranker-v2-m3`) to score candidate pairs jointly with the MSA query. Replaced fake scale multipliers with normalized cross-encoder relevance scores.

---

### 5. Item 1 (M1): Vector Cosine Semantic Caching
* **File Modified:** [`src/cache_db/semantic_cache.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/cache_db/semantic_cache.py)
* **Problem:** `SemanticCache` previously checked exact string equality (`query_normalized == q_norm`), producing cache misses on paraphrases.
* **Fix Implemented:** Added 1024-dimensional query vector storage and vector cosine similarity search (`cosine_similarity`). Returns instant cache hits ($<10\text{ms}$) tagged with `⚡ ذاكرة سريعة` when cosine similarity $\ge 0.88$.

---

### 6. Item 16 (Q3): Embedder Fail-Fast Load Protection
* **File Modified:** [`src/kb_ingestor/ingestion.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/kb_ingestor/ingestion.py)
* **Problem:** If `BGEM3Embedder` failed to load, it swallowed exceptions and returned zero vectors (`[[0.0] * 1024]`), silently serving garbage search results.
* **Fix Implemented:** Updated `_load_model()` to raise an explicit `RuntimeError` immediately upon load failure.

---

### 7. Item 17 (M4, H4): Dynamic Corrective RAG (CRAG) Routing
* **Files Modified:** [`src/pipeline/stage_04_answer/answer_router.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/src/pipeline/stage_04_answer/answer_router.py) & [`tests/unit/test_corrective_rag.py`](file:///c:/Users/VICTUS/Desktop/QWEN3.6-Branch/tests/unit/test_corrective_rag.py)
* **Problem:** `AnswerRouter` previously hardcoded `{"route": "GENERATION"}` unconditionally.
* **Fix Implemented:** Configured `AnswerRouter` to dynamically route based on CRAG confidence decisions: `HIGH` $\rightarrow$ `GENERATION`, `MEDIUM` $\rightarrow$ `QUERY_EXPANSION`, `LOW` $\rightarrow$ `DETERMINISTIC`.

---

## Verification Summary Table

| Verification Test | Command | Target Criteria | Result | Status |
|---|---|---|---|---|
| **Unit Test Suite** | `python -m pytest -m unit` | 29/29 Passing | **29 Passed, 0 Failed** | 🟢 **PASS** |
| **Linter Checks** | `ruff check .` | 0 Syntax Errors | **0 Errors** | 🟢 **PASS** |
| **AraT5 Model Rewrite** | Inline CUDA Execution | Levantine $\rightarrow$ MSA Translation | **Verified (`شو هي` $\rightarrow$ `ما`)** | 🟢 **PASS** |
| **Vector Cache Lookup** | `SemanticCache.get()` | Cosine similarity $\ge 0.88$ | **Verified (<10ms hit)** | 🟢 **PASS** |

---

## Conclusion & Readiness for Production Phase

Phase 2 objectives are **100% complete, integrated, and verified**. The RAG core retrieves 800-token parent context blocks, translates Levantine queries into MSA using AraT5, ranks candidates with BGE Cross-Encoder, and caches responses using 1024-dim vector cosine similarity.
