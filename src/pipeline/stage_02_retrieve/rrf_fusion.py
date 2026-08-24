from typing import Any, Dict, List


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) algorithm combining multi-branch retrieval lists."""

    def __init__(self, k: int = 60):
        self.k = k

    def combine(self, ranked_lists: List[List[Dict[str, Any]]], top_k: int = 15) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        rank_details: Dict[str, Dict[str, int]] = {}

        for list_idx, rank_list in enumerate(ranked_lists):
            list_name = f"branch_{list_idx}"
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc.get("chunk_id") or doc.get("id") or str(hash(doc.get("text", "")[:50]))
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                    scores[doc_id] = 0.0
                    rank_details[doc_id] = {}

                rrf_score = 1.0 / (self.k + rank)
                scores[doc_id] += rrf_score
                rank_details[doc_id][list_name] = rank

        combined_docs = []
        for doc_id, score in scores.items():
            doc = dict(doc_map[doc_id])
            doc["rrf_score"] = round(score, 6)
            doc["ranks"] = rank_details[doc_id]
            combined_docs.append(doc)

        combined_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
        return combined_docs[:top_k]
