import os
import math
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

from core.config import (
    BGE_RERANKER_MODEL_NAME, RERANK_THRESHOLD, RERANK_SCORE_FLOOR,
    RERANK_SCORE_CEIL, DOCUMENT_PRIORITY
)

class RerankerAgent:
    """
    BGE Cross-Encoder Reranker Agent with Post-Threshold Priority Calibration:
    1. Evaluates raw cross-encoder logits for query-chunk pairs.
    2. Drops any chunk below raw logit floor 0.0 (50% sigmoid cutoff) BEFORE priority boost.
    3. Adds calibrated document priority boost to raw logit ONLY for surviving chunks.
    4. Computes Sigmoid similarity percentage (50% to 97%).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerAgent, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if HAS_RERANKER:
            try:
                import torch
                num_cores = os.cpu_count() or 8
                torch.set_num_threads(min(num_cores, 8))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[RerankerAgent] Loading BGE Reranker '{BGE_RERANKER_MODEL_NAME}' on {device}...")
                self.model = CrossEncoder(BGE_RERANKER_MODEL_NAME, max_length=256, device=device)
                print(f"[RerankerAgent] Successfully loaded BGE Reranker on {device}.")
            except Exception as e:
                print(f"[RerankerAgent Warning] Could not load BGE Reranker: {e}")
                self.model = None

    def _get_priority_boost(self, title_or_filename: str) -> float:
        """Returns calibrated additive logit boost for authoritative source documents."""
        for pattern, boost in DOCUMENT_PRIORITY.items():
            if pattern in title_or_filename:
                return boost
        return 0.0

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Reranks candidate chunks with strict threshold cutoff and post-threshold priority boost:
        """
        if not candidate_chunks:
            return []
        if not self.model:
            return candidate_chunks[:top_k]

        # Prepare (query, passage) pairs
        pairs = [[query, chunk.get("text", "")[:600]] for chunk in candidate_chunks]
        raw_scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        surviving_chunks = []
        for i, raw in enumerate(raw_scores):
            raw_logit = float(raw)
            chunk = candidate_chunks[i]
            chunk["raw_rerank_score"] = raw_logit

            # Step 1: Strict Threshold Cutoff (raw logit < 0.0 -> < 50% match)
            if raw_logit < RERANK_THRESHOLD:
                continue  # DISCARD IRRELEVANT CHUNK

            # Step 2: Post-Threshold Priority Boost Calculation
            boost = self._get_priority_boost(chunk.get("title", ""))
            boosted_logit = raw_logit + boost
            chunk["boosted_rerank_score"] = boosted_logit

            # Step 3: Sigmoid Score Conversion
            # sigmoid(0.0) = 50%, sigmoid(2.0) = 88%
            sig = 1.0 / (1.0 + math.exp(-boosted_logit))
            match_pct = int(sig * 100)
            match_pct = min(max(match_pct, RERANK_SCORE_FLOOR), RERANK_SCORE_CEIL)
            chunk["similarity_score"] = match_pct
            
            surviving_chunks.append(chunk)

        # Re-sort surviving chunks by boosted score
        surviving_chunks.sort(key=lambda x: x.get("boosted_rerank_score", 0.0), reverse=True)

        for rank, chunk in enumerate(surviving_chunks[:top_k], 1):
            chunk["rank"] = rank

        return surviving_chunks[:top_k]
