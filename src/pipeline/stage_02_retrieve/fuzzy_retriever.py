from typing import Dict, Any, List
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

class RapidFuzzRetriever:
    """Fuzzy string matching for title & terminology document retrieval."""

    def search(self, query: str, document_titles: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not RAPIDFUZZ_AVAILABLE or not document_titles:
            return []

        results = []
        for doc in document_titles:
            title = doc.get("title", "")
            score = fuzz.token_sort_ratio(query, title)
            if score >= 60:
                doc_copy = dict(doc)
                doc_copy["fuzzy_score"] = float(score)
                results.append(doc_copy)

        results.sort(key=lambda x: x["fuzzy_score"], reverse=True)
        return results[:top_k]
