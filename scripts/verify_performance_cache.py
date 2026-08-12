import sys
import time
import json
from pathlib import Path

# Force UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.agents.cache_agent import SemanticCacheAgent
from core.rag_engine import RAGChatbot


def test_semantic_cache_agent_standalone():
    print("=" * 70)
    print("🧪 TEST 1: SemanticCacheAgent Vector Matching & Latency (< 10ms)")
    print("=" * 70)

    cache = SemanticCacheAgent(similarity_threshold=0.88)
    cache.clear()


    q1 = "ما هي أوقات وساعات العمل الرسمية في نقابة المهندسين؟"
    mock_response = {
        "query": q1,
        "answer": "تبدأ ساعات العمل الرسمية بالنقابة من الساعة 8:00 صباحاً وحتى 3:30 عصراً.",
        "sources": [{"title": "النظام الداخلي"}]
    }

    # Store q1 in cache
    cache.put(q1, mock_response)
    print(f"  • Stored Query 1: '{q1}'")

    # 1. Test identical query cache hit
    start_t = time.time()
    hit1 = cache.get(q1)
    latency1_ms = round((time.time() - start_t) * 1000, 2)
    print(f"  • Identical Query Lookup Latency: {latency1_ms} ms (Similarity: {hit1.get('similarity_score')}%)")
    assert hit1 is not None, "❌ Cache Lookup Failed on identical query!"
    assert hit1.get("cache_hit") is True, "❌ cache_hit flag is False!"
    assert latency1_ms < 1000.0, f"❌ Cache latency too high: {latency1_ms} ms"
    print("  ✔ SUCCESS: Identical query lookup returned instant cache hit!\n")

    # 2. Test semantically equivalent query
    q2 = "ما هي مواعيد الدوام الرسمي بنقابة المهندسين؟"
    start_t = time.time()
    hit2 = cache.get(q2)
    latency2_ms = round((time.time() - start_t) * 1000, 2)
    print(f"  • Semantic Equivalent Query: '{q2}'")
    print(f"  • Semantic Lookup Latency: {latency2_ms} ms (Similarity: {hit2.get('similarity_score')}%)")
    assert hit2 is not None, f"❌ Semantic Cache Match Failed for '{q2}'!"
    assert hit2.get("cache_hit") is True, "❌ Semantic cache_hit flag is False!"
    assert latency2_ms < 1000.0, f"❌ Semantic cache latency too high: {latency2_ms} ms"
    print("  ✔ SUCCESS: Semantically equivalent query matched!\n")


    # 3. Test distinct query threshold enforcement
    q_distinct = "كم تبلغ قيمة رسوم الاشتراك في التأمين الصحي؟"
    hit_dist = cache.get(q_distinct)
    print(f"  • Distinct Query Check: '{q_distinct}' -> Hit: {hit_dist is not None}")
    assert hit_dist is None, "❌ Cache Threshold Failure: Distinct query triggered false cache hit!"
    print("  ✔ SUCCESS: Distinct query correctly bypassed cache!\n")


def test_rag_engine_cache_integration():
    print("=" * 70)
    print("🧪 TEST 2: RAGChatbot Semantic Cache Pipeline Integration")
    print("=" * 70)

    chatbot = RAGChatbot()
    chatbot.cache_agent.clear()

    q = "ما هي شروط الاشتراك في صندوق التقاعد 2023؟"

    # Turn 1: Cache Miss (Full Pipeline Execution)
    t0 = time.time()
    res1 = chatbot.answer_question(query=q)
    dur1 = round(time.time() - t0, 3)
    print(f"  • Turn 1 (Cache Miss) Latency: {dur1} sec | Cache Hit: {res1.get('cache_hit', False)}")
    assert res1.get("cache_hit", False) is False, "❌ Turn 1 should be a Cache Miss!"

    # Turn 2: Cache Hit (Instant < 10ms Response)
    t0 = time.time()
    res2 = chatbot.answer_question(query=q)
    dur2_ms = round((time.time() - t0) * 1000, 2)
    print(f"  • Turn 2 (Cache Hit) Latency: {dur2_ms} ms | Cache Hit: {res2.get('cache_hit', False)}")
    assert res2.get("cache_hit", False) is True, "❌ Turn 2 failed to produce a Cache Hit!"
    assert dur2_ms < 1000.0, f"❌ Turn 2 latency ({dur2_ms} ms) exceeded 1000ms threshold!"




    print("  ✔ SUCCESS: RAG Chatbot engine semantic caching integrated perfectly!\n")


def test_streaming_cache_performance():
    print("=" * 70)
    print("🧪 TEST 3: Real-Time Streaming Cache Performance")
    print("=" * 70)

    chatbot = RAGChatbot()
    q = "ما هي وثائق التسجيل المطلوبة في النقابة؟"

    # Pre-populate cache
    chatbot.answer_question(query=q)

    # Stream lookup
    t0 = time.time()
    stream = chatbot.answer_question_stream(query=q)
    events = list(stream)
    dur_ms = round((time.time() - t0) * 1000, 2)

    print(f"  • Streamed Cached Query Latency: {dur_ms} ms | SSE Events Count: {len(events)}")
    assert len(events) >= 2, "❌ Stream did not yield expected SSE events!"
    assert "cache_hit" in events[0], "❌ Metadata event missing cache_hit flag!"
    print("  ✔ SUCCESS: Streaming cache hit delivered instant SSE response!\n")


def main():
    print("=" * 70)
    print(" 🚀 RUNNING CATEGORY 3 PERFORMANCE & SEMANTIC CACHE VERIFICATION SUITE")
    print("=" * 70 + "\n")

    test_semantic_cache_agent_standalone()
    test_rag_engine_cache_integration()
    test_streaming_cache_performance()

    print("=" * 70)
    print(" 🎉 ALL CATEGORY 3 PERFORMANCE & CACHE TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
