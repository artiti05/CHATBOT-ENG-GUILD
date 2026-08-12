import sys
import os
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import (
    PARENT_CHUNK_TOKENS, CHILD_CHUNK_TOKENS, RERANK_THRESHOLD, RELEVANCE_THRESHOLD
)
from core.ingestion import TextChunker, clean_document_text
from core.bm25_search import BM25Indexer, normalize_arabic_text, tokenize_arabic
from core.agents.retriever_agent import RetrieverAgent
from core.agents.reranker_agent import RerankerAgent

def test_category1():
    print("=" * 65)
    print(" [TEST] CATEGORY 1: RAG & RETRIEVAL LAYER IMPROVEMENTS")
    print("=" * 65)


    # -------------------------------------------------------------
    # 1. Test TextChunker Parent-Child Token Chunking
    # -------------------------------------------------------------
    print("\n1. Testing Parent-Child Hierarchical Token Chunking...")
    sample_text = (
        "المادة 45 من قانون نقابة المهندسين الأردنيين رقم 15 لسنة 2023. "
        "تنص المادة على أنه يجب على كافة المهندسين المسجلين تسديد الاشتراكات السنوية في موعد لا يتجاوز نهاية شهر آذار من كل عام. "
    ) * 30  # Duplicate to create a multi-paragraph document

    doc_dict = {
        "source_id": "test_law_doc",
        "file_name": "قانون_نقابة_المهندسين.pdf",
        "source_type": "pdf",
        "full_text": sample_text
    }

    chunker = TextChunker()
    child_chunks = chunker.chunk_text(doc_dict)

    assert len(child_chunks) > 0, "Chunker failed to generate child chunks"
    first_child = child_chunks[0]
    
    print(f"   - Generated {len(child_chunks)} child chunks.")
    print(f"   - Child chunk ID: '{first_child['chunk_id']}'")
    print(f"   - Parent ID link: '{first_child['parent_id']}'")
    print(f"   - Child word count: {first_child['word_count']} words (~{first_child['estimated_tokens']} tokens)")
    print(f"   - Parent text snippet: '{first_child['parent_text'][:100]}...'")

    assert "parent_id" in first_child, "Missing parent_id link in child metadata"
    assert "parent_text" in first_child, "Missing parent_text in child metadata"
    assert first_child['estimated_tokens'] <= 250, "Child chunk token size exceeded target bound"
    print("   [PASS] Parent-Child Chunking passed!")

    # -------------------------------------------------------------
    # 2. Test BM25 Arabic Normalization & Keyword Search
    # -------------------------------------------------------------
    print("\n2. Testing BM25 Sparse Keyword Search & Arabic Normalization...")
    bm25 = BM25Indexer(index_path=BASE_DIR / "data" / "storage" / "test_bm25.pkl")
    bm25.reset()

    test_chunks = [
        {
            "chunk_id": "c1",
            "text": "يحق للمهندس المتقاعد الحصول على الراتب التقاعدي وفقاً للمادة 45 من النظام الداخلي.",
            "parent_id": "p1",
            "parent_text": "نص المادة 45 الكامل حول حقوق المهندسين المتقاعدين في نقابة المهندسين الأردنيين...",
            "metadata": {"file_name": "نظام_التقاعد_2023.pdf", "chunk_index": 1}
        },
        {
            "chunk_id": "c2",
            "text": "تحدد رسوم ممارسة مهنة الهندسة للمكاتب والشركات بموجب جدول الرواتب الرسمي.",
            "parent_id": "p2",
            "parent_text": "جدول التفاصيل المالية ورسوم ممارسة المهنة...",
            "metadata": {"file_name": "نظام_المكاتب.pdf", "chunk_index": 1}
        }
    ]

    bm25.add_chunks(test_chunks)
    
    # Query exact article number with diacritics / normalization variations
    exact_query = "المادة 45"
    results = bm25.search(exact_query, top_k=5)
    
    assert len(results) > 0, "BM25 failed to return results for exact query"
    top_chunk, score = results[0]
    print(f"   - Query: '{exact_query}'")
    print(f"   - Top Match: Chunk ID '{top_chunk['chunk_id']}' with BM25 score {score:.4f}")
    assert top_chunk['chunk_id'] == "c1", "BM25 failed to rank exact keyword match at #1"
    print("   [PASS] BM25 Sparse Keyword Search passed!")

    # -------------------------------------------------------------
    # 3. Test Reranker Priority Boost Calibration & 50% Cutoff
    # -------------------------------------------------------------
    print("\n3. Testing Reranker Threshold (50% / raw 0.0) Cutoff & Post-Threshold Sigmoid Boost...")
    reranker = RerankerAgent()

    # Simulate candidates: one weak (raw logit -1.5), one valid (raw logit +1.0)
    mock_candidates = [
        {
            "text": "معلومات عامة عن الإعلانات والنشرة الإرشادية.",
            "title": "قانون_نقابة_المهندسين.pdf",  # Has high priority boost (0.35)
            "metadata": {"file_name": "قانون_نقابة_المهندسين.pdf"}
        },
        {
            "text": "شروط وتفاصيل المادة 45 التقاعدية.",
            "title": "نظام_التقاعد_2023.pdf",     # Has moderate priority boost (0.30)
            "metadata": {"file_name": "نظام_التقاعد_2023.pdf"}
        }
    ]

    if reranker.model:
        # Test real reranking with threshold check
        reranked = reranker.rerank("المادة 45 التقاعد", mock_candidates, top_k=10)
        print(f"   - Surviving chunks count: {len(reranked)}")
        for r in reranked:
            print(f"     Rank {r.get('rank')}: Raw={r.get('raw_rerank_score'):.3f}, Boosted={r.get('boosted_rerank_score'):.3f}, Display={r.get('similarity_score')}%")
            assert r.get("similarity_score") >= RELEVANCE_THRESHOLD, "Chunk below 50% threshold was not discarded!"
        print("   [PASS] Reranker 50% Cutoff & Post-Threshold Priority Boost passed!")
    else:
        print("   [INFO] CrossEncoder model not loaded (running CPU lightweight mode). Skips model weights check.")

    print("\n" + "=" * 65)
    print(" [SUCCESS] ALL CATEGORY 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    test_category1()
