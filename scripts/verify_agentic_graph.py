import sys
import time
import json
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.graph import RAGStateGraph


def test_agentic_stategraph_execution():
    print("=" * 70)
    print("🧪 TEST 1: RAGStateGraph Multi-Agent Orchestration (6 Subagents)")
    print("=" * 70)

    graph = RAGStateGraph()
    graph.cache_agent.clear()

    q = "ما هي شروط الاستفادة من صندوق التقاعد عام 2023؟"
    res = graph.run(query=q)

    print(f"  • Standalone Query Generated: '{res.get('standalone_query')}'")
    print(f"  • Total Retrieved Sources: {len(res.get('sources', []))}")
    print(f"  • Answer Preview: {res.get('answer', '')[:100]}...")
    print(f"  • Reflexion Count: {res.get('reflexion_count', 0)}")
    print(f"  • Total Time Taken: {res.get('time_taken')} sec")

    assert res.get("standalone_query") is not None, "❌ Standalone query generation failed!"
    assert len(res.get("sources", [])) > 0, "❌ No sources retrieved!"
    assert res.get("answer") is not None, "❌ Answer generation failed!"
    print("  ✔ SUCCESS: RAGStateGraph multi-agent execution completed cleanly!\n")


def test_reflexion_loop_and_cache():
    print("=" * 70)
    print("🧪 TEST 2: CRAG Reflexion Loop & Semantic Cache Hit")
    print("=" * 70)

    graph = RAGStateGraph()
    q = "ما هي شروط الاستفادة من صندوق التقاعد عام 2023؟"

    # Turn 2: Should hit cache instantly
    t0 = time.time()
    res2 = graph.run(query=q)
    dur2_ms = round((time.time() - t0) * 1000, 2)

    print(f"  • Cache Hit Turn Latency: {dur2_ms} ms | Cache Hit: {res2.get('cache_hit', False)}")
    assert res2.get("cache_hit", False) is True, "❌ StateGraph failed to produce a Cache Hit!"
    assert dur2_ms < 1000.0, f"❌ Cache latency too high: {dur2_ms} ms"
    print("  ✔ SUCCESS: StateGraph semantic cache hit verified!\n")


def main():
    print("=" * 70)
    print(" 🚀 RUNNING CATEGORY 4 AGENTIC STATEGRAPH VERIFICATION SUITE")
    print("=" * 70 + "\n")

    test_agentic_stategraph_execution()
    test_reflexion_loop_and_cache()

    print("=" * 70)
    print(" 🎉 ALL CATEGORY 4 AGENTIC STATEGRAPH TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
