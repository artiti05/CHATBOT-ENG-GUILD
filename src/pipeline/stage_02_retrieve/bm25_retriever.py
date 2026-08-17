from typing import List, Dict, Any
from .bm25_search import BM25Indexer

class BM25Retriever:
    """Wrapper around BM25Indexer for retrieving keyword matches as unified document dicts."""

    def __init__(self):
        self.indexer = BM25Indexer()

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        raw_results = self.indexer.search(query, top_k=top_k)
        flattened = []
        for doc_item, score in raw_results:
            if isinstance(doc_item, dict):
                item_copy = dict(doc_item)
                item_copy["bm25_score"] = float(score)
                item_copy["similarity_score"] = float(score)
                flattened.append(item_copy)
        return flattened
