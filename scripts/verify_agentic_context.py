import sys
import json
from pathlib import Path

# Force UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.agents.rewriter_agent import DialectRewriterAgent
from core.agents.verifier_agent import VerifierAgent, normalize_arabic_stem
from core.rag_engine import RAGChatbot


def test_multi_turn_query_rewriting():
    print("=" * 70)
    print("🧪 TEST 1: Multi-Turn Conversation Memory & Query Rewriting")
    print("=" * 70)

    rewriter = DialectRewriterAgent()
    history = [
        {"role": "user", "content": "ما هي شروط الاشتراك في صندوق التقاعد 2023؟"},
        {"role": "assistant", "content": "شروط الاشتراك تتطلب أن يكون المهندس مسجلاً بالنقابة ومسدداً لالتزاماته السنوية."}
    ]

    turn2_query = "وما هي الرسوم والشرائح المستحقة عليه؟"
    rewritten_q, meta = rewriter.rewrite_query(turn2_query, history=history)

    print(f"  • Turn 1: {history[0]['content']}")
    print(f"  • Turn 2 (Dependent Query): '{turn2_query}'")
    print(f"  • Rewritten Standalone Query: '{rewritten_q}'")
    print(f"  • Rewriter Metadata: {meta}")

    # Verification checks
    assert "تقاعد" in rewritten_q or "التقاعد" in rewritten_q or "صندوق" in rewritten_q, \
        "❌ Multi-Turn Rewrite Failed: Subject 'صندوق التقاعد' was lost from conversation memory!"
    print("  ✔ SUCCESS: Multi-Turn Conversation Memory correctly preserved subject context!\n")


def test_dialect_to_msa_translation():
    print("=" * 70)
    print("🧪 TEST 2: Dynamic Dialect-to-MSA Legal Translation")
    print("=" * 70)

    rewriter = DialectRewriterAgent()

    dialect_queries = [
        ("قديش بدفع اشتراك التأمين الصحي عائلات؟", ["رسوم", "التأمين الصحي"]),
        ("شو الأوراق المطلوبة عشان أسجل بالنقابة؟", ["الوثائق", "المستندات", "التسجيل"]),
        ("ازاي بقدر استعلم عن قرض الاثاث؟", ["قرض", "الأثاث"])
    ]

    for raw_dialect, expected_keywords in dialect_queries:
        rewritten_q, meta = rewriter.rewrite_query(raw_dialect)
        print(f"  • Raw Dialect Input: '{raw_dialect}'")
        print(f"  • MSA Rewritten Query: '{rewritten_q}' (Method: {meta.get('method')})")

        found = any(kw in rewritten_q for kw in expected_keywords)
        assert found, f"❌ Dialect Translation Failed for '{raw_dialect}'! Rewritten: '{rewritten_q}'"
        print(f"  ✔ Validated translation for '{raw_dialect}'\n")

    print("  ✔ SUCCESS: All dialect queries correctly normalized to MSA legal terms!\n")


def test_crag_confidence_and_expansions():
    print("=" * 70)
    print("🧪 TEST 3: Corrective RAG (CRAG) Confidence & Query Expansions")
    print("=" * 70)

    verifier = VerifierAgent(confidence_threshold=55.0)

    # Case A: Low confidence retrieval
    low_sources = [{"similarity_score": 42.0}, {"similarity_score": 38.0}]
    crag_eval_a = verifier.evaluate_retrieval_confidence(low_sources)
    print(f"  • Low Confidence Case Evaluation: {crag_eval_a}")
    assert crag_eval_a["confidence_low"] is True, "❌ CRAG Evaluation Failed: Low confidence not detected!"

    # Case B: Query expansions for fallback search
    query = "شروط الاستفادة من منفعة الزواج للمهندسين الشباب"
    expansions = verifier.generate_query_expansions(query)
    print(f"  • Original Query: '{query}'")
    print(f"  • Generated CRAG Expansions: {expansions}")
    assert len(expansions) > 0, "❌ CRAG Query Expansion Failed: No expanded queries generated!"
    print("  ✔ SUCCESS: Corrective RAG confidence evaluation & query expansions verified!\n")


def test_answer_groundedness_verification():
    print("=" * 70)
    print("🧪 TEST 4: Answer Groundedness & Citation Verification")
    print("=" * 70)

    verifier = VerifierAgent()

    sample_sources = [
        {
            "rank": 1,
            "title": "نظام التقاعد 2023",
            "text": "يصرف للمهندس المتقاعد راتب تقاعدي شهر بواقع 300 دينار للشريحة الأولى و 500 دينار للشريحة الثانية عند إتمام السن القانونية."
        }
    ]

    # Grounded answer with citation
    grounded_answer = "يبلغ الراتب التقاعدي للشريحة الأولى 300 دينار وللشريحة الثانية 500 دينار وفق نظام التقاعد [المصدر 1]."
    verification_good = verifier.verify_answer_groundedness(grounded_answer, sample_sources)
    print(f"  • Grounded Answer Check: Score = {verification_good['score']}%, Grounded = {verification_good['is_grounded']}")
    assert verification_good["is_grounded"] is True, "❌ Groundedness Verification Failed on valid grounded answer!"

    # Ungrounded / Hallucinated answer
    hallucinated_answer = "تمنح النقابة رحلات طيران مجانية وسيارات حديثة لجميع المهندسين الخريجين."
    verification_bad = verifier.verify_answer_groundedness(hallucinated_answer, sample_sources)
    print(f"  • Hallucinated Answer Check: Score = {verification_bad['score']}%, Grounded = {verification_bad['is_grounded']}")
    assert verification_bad["is_grounded"] is False, "❌ Groundedness Verification Failed: Hallucination was not caught!"

    print("  ✔ SUCCESS: Answer groundedness & citation verification functioning perfectly!\n")


def main():
    print("=" * 70)
    print(" 🚀 RUNNING CATEGORY 2 VERIFICATION SUITE")
    print("   (Multi-Turn Rewriting, Dialect Translation, CRAG & Groundedness)")
    print("=" * 70 + "\n")

    test_multi_turn_query_rewriting()
    test_dialect_to_msa_translation()
    test_crag_confidence_and_expansions()
    test_answer_groundedness_verification()

    print("=" * 70)
    print(" 🎉 ALL CATEGORY 2 TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
