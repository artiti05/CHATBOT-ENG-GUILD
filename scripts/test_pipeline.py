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


def run_pipeline_benchmark():
    print("=" * 75)
    print(" 🚀 RUNNING MULTI-AGENT PIPELINE REGRESSION BENCHMARK SUITE")
    print("=" * 75 + "\n")

    graph = RAGStateGraph()
    graph.cache_agent.clear()

    benchmark_queries = [
        ("مواعيد الدوام الرسمي بالنقابة", "MSA Organizational Query"),
        ("شو الأوراق المطلوبة عشان أسجل بالنقابة؟", "Jordanian Dialect Query"),
        ("ما هي شروط الاشتراك في صندوق التقاعد 2023؟", "Legal / Financial Query"),
    ]

    results = []

    for idx, (q, q_type) in enumerate(benchmark_queries, 1):
        print(f"[{idx}/{len(benchmark_queries)}] Testing ({q_type}): '{q}'")

        # Cold Run (Cache Miss)
        t0 = time.time()
        res_cold = graph.run(query=q)
        dur_cold = round(time.time() - t0, 3)

        # Warm Run (Cache Hit)
        t1 = time.time()
        res_warm = graph.run(query=q)
        dur_warm_ms = round((time.time() - t1) * 1000, 2)

        record = {
            "query": q,
            "type": q_type,
            "standalone_query": res_cold.get("standalone_query"),
            "sources_count": len(res_cold.get("sources", [])),
            "cold_latency_sec": dur_cold,
            "warm_cache_latency_ms": dur_warm_ms,
            "cache_hit_verified": res_warm.get("cache_hit", False),
            "reflexion_count": res_cold.get("reflexion_count", 0)
        }
        results.append(record)

        print(f"    • Standalone MSA: '{record['standalone_query']}'")
        print(f"    • Cold Latency: {dur_cold}s | Warm Cache Latency: {dur_warm_ms}ms")
        print(f"    • Sources Found: {record['sources_count']} | Cache Verified: {record['cache_hit_verified']}\n")

    print("=" * 75)
    print(" 📊 PIPELINE BENCHMARK SUMMARY REPORT")
    print("=" * 75)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("=" * 75)
    print(" 🎉 ALL PIPELINE BENCHMARK TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_pipeline_benchmark()
