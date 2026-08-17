from typing import Dict, Any, List
from src.config import DOCUMENT_PRIORITY, RERANK_THRESHOLD

class PriorityReranker:
    """Reranks candidates using Cross-Encoder or raw logit scores and applies document priority boosts."""

    def __init__(self, priority_map: Dict[str, float] = DOCUMENT_PRIORITY):
        self.priority_map = priority_map

    def apply_priority_boost(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        boosted = []
        for cand in candidates:
            c = dict(cand)
            title = c.get("title", "")
            boost = 0.0
            for key, val in self.priority_map.items():
                if key in title:
                    boost = max(boost, val)
                    break

            if "distance" in c and c["distance"] is not None:
                base_sim = max(0.0, min(1.0, 1.0 - float(c["distance"]))) * 100.0
            elif "similarity_score" in c and float(c.get("similarity_score", 0)) > 1.0:
                base_sim = float(c["similarity_score"])
            elif "rrf_score" in c and c["rrf_score"] is not None:
                rrf_val = float(c["rrf_score"])
                base_sim = min(92.0, max(55.0, rrf_val * 2400.0))
            else:
                base_sim = 60.0

            final_score = round(min(98.0, max(50.0, base_sim + (boost * 20.0))), 1)
            c["priority_boost"] = boost
            c["final_score"] = final_score
            c["similarity_score"] = final_score
            boosted.append(c)

        boosted.sort(key=lambda x: x["similarity_score"], reverse=True)
        return boosted
