from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from .ingestion import BGEM3Embedder
from .bm25_search import BM25Indexer

class RetrieverAgent:
    """
    Hybrid Knowledge Retriever Agent combining:
    1. Dense Vector Search via ChromaDB & BGE-M3.
    2. Sparse Keyword Search via BM25 (Arabic Normalized).
    3. Reciprocal Rank Fusion (RRF k=60) score merging.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.embedder = BGEM3Embedder()
        self.bm25 = BM25Indexer()

    def retrieve_hybrid(self, query: str, top_k: int = 20, rrf_k: int = 60,
                        query_vec: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        - Fetches top_k candidates from ChromaDB Dense Search.
        - Fetches top_k candidates from BM25 Sparse Search.
        - Dense (GPU) and BM25 (CPU) run CONCURRENTLY.
        - Fuses rankings using RRF formula.
        query_vec: optional precomputed BGE-M3 dense vector for the query
        (e.g. reused from the semantic cache check) to avoid re-embedding.
        Returns unified list of top candidate chunks.
        """
        if not query or not query.strip():
            return []

        def _dense_search() -> List[Dict[str, Any]]:
            dense_candidates: List[Dict[str, Any]] = []
            try:
                if query_vec is not None:
                    qv = query_vec
                else:
                    dense_vecs, _ = self.embedder.embed_texts([query])
                    qv = dense_vecs[0] if dense_vecs else None
                if qv is not None:
                    results = self.collection.query(
                        query_embeddings=[qv],
                        n_results=min(top_k * 2, 30),
                        include=["documents", "metadatas", "distances"]
                    )
                    if results and results.get("documents") and results["documents"][0]:
                        docs = results["documents"][0]
                        metas = results["metadatas"][0]
                        dists = results["distances"][0]
                        ids = results["ids"][0] if "ids" in results else [f"dense_{i}" for i in range(len(docs))]

                        for i in range(len(docs)):
                            meta = metas[i] if i < len(metas) else {}
                            dense_candidates.append({
                                "chunk_id": ids[i],
                                "text": docs[i],
                                "title": meta.get("file_name", "وثيقة"),
                                "section_title": f"جزء {meta.get('chunk_index', 1)}",
                                "distance": dists[i],
                                "parent_id": meta.get("parent_id", ""),
                                "parent_text": meta.get("parent_text", docs[i]),
                                "metadata": meta
                            })
            except Exception as e:
                print(f"[Retriever Warning] Dense search failed: {e}")
            return dense_candidates

        def _sparse_search() -> List[Dict[str, Any]]:
            bm25_candidates: List[Dict[str, Any]] = []
            try:
                bm25_raw_results = self.bm25.search(query, top_k=top_k)
                for chunk_item, score in bm25_raw_results:
                    meta = chunk_item.get("metadata", {})
                    bm25_candidates.append({
                        "chunk_id": chunk_item.get("chunk_id", f"bm25_{len(bm25_candidates)}"),
                        "text": chunk_item.get("text", ""),
                        "title": meta.get("file_name", "وثيقة"),
                        "section_title": f"جزء {meta.get('chunk_index', 1)}",
                        "bm25_score": score,
                        "parent_id": chunk_item.get("parent_id", ""),
                        "parent_text": chunk_item.get("parent_text", chunk_item.get("text", "")),
                        "metadata": meta
                    })
            except Exception as e:
                print(f"[Retriever Warning] BM25 search failed: {e}")
            return bm25_candidates

        # Run dense (GPU-bound) and BM25 (CPU-bound) in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(_dense_search)
            sparse_future = pool.submit(_sparse_search)
            dense_candidates = dense_future.result()
            bm25_candidates = sparse_future.result()

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, cand in enumerate(dense_candidates, start=1):
            cid = cand.get("chunk_id") or cand.get("text")
            rrf_map[cid] = cand
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        for rank, cand in enumerate(bm25_candidates, start=1):
            cid = cand.get("chunk_id") or cand.get("text")
            if cid not in rrf_map:
                rrf_map[cid] = cand
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        merged_results = []
        for cid in sorted_ids:
            chunk_obj = rrf_map[cid]
            chunk_obj["rrf_score"] = rrf_scores[cid]
            merged_results.append(chunk_obj)

        return merged_results
