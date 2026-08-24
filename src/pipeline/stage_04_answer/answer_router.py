from typing import Any, Dict


class AnswerRouter:
    """Routes queries dynamically to GENERATION, QUERY_EXPANSION, or DETERMINISTIC fallback based on CRAG confidence."""

    def route(self, query_obj: Dict[str, Any], confidence_res: Dict[str, Any]) -> Dict[str, Any]:
        decision = confidence_res.get("decision", "HIGH")
        if decision == "LOW":
            return {
                "route": "DETERMINISTIC",
                "llm_required": False,
                "reason": "crag_low_confidence_fallback"
            }
        elif decision == "MEDIUM":
            return {
                "route": "QUERY_EXPANSION",
                "llm_required": True,
                "reason": "crag_medium_confidence_expansion"
            }
        return {
            "route": "GENERATION",
            "llm_required": True,
            "reason": "crag_high_confidence_generation"
        }
