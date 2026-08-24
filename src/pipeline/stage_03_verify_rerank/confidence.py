from typing import Any, Dict, List


class DeterministicConfidenceEvaluator:
    """Evaluates retrieval confidence deterministically to skip unnecessary LLM verifiers."""

    @staticmethod
    def evaluate(candidates: List[Dict[str, Any]], query_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not candidates:
            return {
                "score": 0.0,
                "decision": "LOW",
                "reason": "no_candidates",
                "llm_verifier_used": False
            }

        top_score = candidates[0].get("final_score", 0.0)
        has_intent = len(query_obj.get("intent", [])) > 0
        has_priority_boost = candidates[0].get("priority_boost", 0.0) > 0.0

        confidence = top_score
        if has_intent:
            confidence += 5.0
        if has_priority_boost:
            confidence += 5.0

        confidence_val = min(0.98, round(confidence / 100.0, 2))

        if confidence_val >= 0.70:
            decision = "HIGH"
        elif confidence_val >= 0.50:
            decision = "MEDIUM"
        else:
            decision = "LOW"

        return {
            "score": confidence_val,
            "decision": decision,
            "top_score": top_score,
            "has_intent": has_intent,
            "llm_verifier_used": False
        }
