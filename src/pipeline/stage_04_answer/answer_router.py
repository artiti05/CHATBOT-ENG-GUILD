from typing import Dict, Any, List

class AnswerRouter:
    """Routes queries to TICKETING, DETERMINISTIC, or GENERATION (RAG LLM)."""

    def route(self, query_obj: Dict[str, Any], confidence_res: Dict[str, Any]) -> Dict[str, Any]:
        intents = query_obj.get("intent", [])
        is_explicit_ticket = query_obj.get("is_explicit_ticket", False)

        if is_explicit_ticket or "TICKETING" in intents:
            return {
                "route": "TICKETING",
                "llm_required": False,
                "reason": "explicit_support_ticket_request"
            }

        return {
            "route": "GENERATION",
            "llm_required": True,
            "reason": "dynamic_rag_synthesis"
        }
