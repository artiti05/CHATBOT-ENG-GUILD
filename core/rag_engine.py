import os
import re
import math
import time
import requests
from typing import Dict, Any, List, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

from core.config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, BGE_RERANKER_MODEL_NAME,
    OLLAMA_URL, OLLAMA_CHAT_MODEL, OLLAMA_VISION_MODEL,
    NUM_CTX, NUM_PREDICT, OLLAMA_KEEP_ALIVE,
    RERANK_THRESHOLD, RELEVANCE_THRESHOLD, RERANK_SCORE_FLOOR, RERANK_SCORE_CEIL,
    DOCUMENT_PRIORITY
)

from core.ingestion import BGEM3Embedder
from core.agents.retriever_agent import RetrieverAgent
from core.agents.reranker_agent import RerankerAgent
from core.agents.rewriter_agent import DialectRewriterAgent
from core.agents.verifier_agent import VerifierAgent
from core.agents.cache_agent import SemanticCacheAgent



# ---------------------------------------------------------------------------
# 1. BGE CROSS-ENCODER RERANKER LAYER
# ---------------------------------------------------------------------------
class BGEReranker:
    """Delegates to core.agents.reranker_agent.RerankerAgent"""
    def __init__(self):
        self.agent = RerankerAgent()

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        return self.agent.rerank(query, candidate_chunks, top_k=top_k)


# ---------------------------------------------------------------------------
# 2. HYBRID VECTOR SEARCH KNOWLEDGE RETRIEVER
# ---------------------------------------------------------------------------
class KnowledgeRetriever:
    def __init__(self):
        self.retriever_agent = RetrieverAgent()
        self.reranker_agent = RerankerAgent()

    def retrieve(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []

        # Step 1: Hybrid Retrieval (Dense Vector + Arabic BM25 + Reciprocal Rank Fusion RRF k=60)
        candidates = self.retriever_agent.retrieve_hybrid(query_text, top_k=min(top_k * 2, 30))

        # Step 2: Cross-Encoder Reranking with Strict Cutoff (raw logit < 0.0 -> < 50% dropped) & Post-Threshold Priority Boost
        reranked = self.reranker_agent.rerank(query_text, candidates, top_k=top_k)

        # Step 3: Hierarchical Parent-Child Context Mapping
        for chunk in reranked:
            if "parent_text" in chunk and chunk["parent_text"]:
                chunk["child_text"] = chunk.get("text", "")
                chunk["text"] = chunk["parent_text"]  # Provide full ~800 token parent context block to LLM

        return reranked


def clean_formatting(text: str) -> str:
    """Strips markdown bold/italic asterisks (**text**) and header hashes to deliver clean presentation text."""
    if not text:
        return ""
    cleaned = re.sub(r'\*+', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


from core.graph import RAGStateGraph


# ---------------------------------------------------------------------------
# 3. CONSOLIDATED RAG & DIALECT CHATBOT ENGINE (StateGraph Coordinator)
# ---------------------------------------------------------------------------
class RAGChatbot:
    def __init__(self, provider: str = "ollama"):
        self.graph = RAGStateGraph()
        self.cache_agent = self.graph.cache_agent
        self.retriever = self.graph.retriever_agent
        self.rewriter = self.graph.rewriter_agent
        self.verifier = self.graph.verifier_agent
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_CHAT_MODEL
        self.provider = provider.lower()

    def detect_and_normalize_query(self, query: str) -> Tuple[str, str]:
        return self.graph._detect_accent(query)

    def answer_question(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 15) -> Dict[str, Any]:
        return self.graph.run(query=query, history=history, top_k=top_k)

    def answer_question_stream(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 15):
        return self.graph.run_stream(query=query, history=history, top_k=top_k)




