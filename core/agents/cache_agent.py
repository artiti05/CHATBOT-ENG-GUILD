import os
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from core.config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_ENTRIES, CACHE_PERSIST_PATH
from core.ingestion import BGEM3Embedder


def cosine_similarity_vec(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vector lists."""
    arr1 = np.array(v1, dtype=np.float32)
    arr2 = np.array(v2, dtype=np.float32)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


class SemanticCacheAgent:
    """
    Agentic Semantic Query Cache Agent.
    Interceptors incoming user queries, calculates BGE-M3 query dense vectors,
    and performs cosine similarity checks against in-memory cached vectors.
    Returns pre-computed RAG answers in < 10ms for semantically equivalent queries (similarity >= 0.95).
    """

    def __init__(self,
                 similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
                 max_entries: int = CACHE_MAX_ENTRIES,
                 persist_path: Path = CACHE_PERSIST_PATH):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.persist_path = Path(persist_path)
        self.embedder = BGEM3Embedder()
        self.cache: List[Dict[str, Any]] = []
        self.last_query_vec: Optional[List[float]] = None
        self._load_cache()
        # Warmup embedder model once on init to eliminate PyTorch first-run JIT latency
        if self.embedder.model:
            _ = self.embedder.embed_single_text("warmup")


    def _load_cache(self) -> None:
        """Loads cached entries from JSON file if present."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.cache = data[:self.max_entries]
            except Exception as e:
                self.cache = []

    def save_cache(self) -> None:
        """Persists cached entries to JSON file."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Interceptors incoming query.
        Returns cached response dict if cosine similarity >= 0.95, else returns None.
        """
        q = query.strip()
        # Always compute (and expose) the query vector so downstream retrieval
        # can reuse it instead of re-embedding the same query.
        self.last_query_vec: Optional[List[float]] = None
        if not q:
            return None

        start_time = time.time()
        # Compute dense embedding vector for incoming query
        query_vec = self.embedder.embed_single_text(q)
        if not query_vec:
            return None
        self.last_query_vec = query_vec

        if not self.cache:
            return None

        # Vectorized cosine scan: single matrix op instead of a Python loop
        q_arr = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm == 0:
            return None

        entries = [e for e in self.cache if e.get("vector")]
        if not entries:
            return None
        mat = np.array([e["vector"] for e in entries], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1e-9
        sims = (mat @ q_arr) / (norms * q_norm)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_entry = entries[best_idx]

        if best_score >= self.similarity_threshold and best_entry:
            elapsed = round((time.time() - start_time) * 1000, 2)  # In milliseconds
            cached_res = dict(best_entry.get("response", {}))
            cached_res["query"] = query  # Update to current user query
            cached_res["cache_hit"] = True
            cached_res["similarity_score"] = round(best_score * 100, 1)
            cached_res["time_taken_ms"] = elapsed
            return cached_res

        return None

    def put(self, query: str, response: Dict[str, Any]) -> None:
        """Stores query, embedding vector, and response payload in semantic cache."""
        q = query.strip()
        if not q or not response:
            return

        query_vec = self.embedder.embed_single_text(q)
        if not query_vec:
            return

        # Check if already exists with high similarity to update
        for entry in self.cache:
            if cosine_similarity_vec(query_vec, entry.get("vector", [])) >= 0.98:
                entry["query"] = q
                entry["response"] = response
                self.save_cache()
                return

        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_entries:
            self.cache.pop(0)

        self.cache.append({
            "query": q,
            "vector": query_vec,
            "response": response,
            "timestamp": time.time()
        })
        self.save_cache()

    def clear(self) -> None:
        """Clears all cached entries."""
        self.cache = []
        if self.persist_path.exists():
            try:
                self.persist_path.unlink()
            except Exception:
                pass
