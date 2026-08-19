from typing import Dict, Any, List

class AnswerRouter:
    """Routes queries to DETERMINISTIC or GENERATION (RAG LLM)."""

    def route(self, query_obj: Dict[str, Any], confidence_res: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "route": "GENERATION",
            "llm_required": True,
            "reason": "dynamic_rag_synthesis"
        }
