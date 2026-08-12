import os
import re
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.config import BM25_INDEX_PATH

def normalize_arabic_text(text: str) -> str:
    """
    Normalizes Arabic text for BM25 keyword matching:
    1. Removes Arabic diacritics (Tashkeel).
    2. Removes Tatweel (Kashida).
    3. Normalizes Alef forms (أ, إ, آ -> ا).
    4. Normalizes Yaa (ى -> ي) and Teh Marbuta (ة -> ه).
    5. Strips non-alphanumeric/punctuation artifacts while keeping numbers intact.
    """
    if not text:
        return ""

    # Remove Tashkeel
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    # Remove Tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize Alef
    text = re.sub(r'[إأآ]', 'ا', text)
    # Normalize Yaa & Teh Marbuta
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    
    # Lowercase English if any
    text = text.lower()
    return text

def tokenize_arabic(text: str) -> List[str]:
    """Tokenizes normalized Arabic/English text into words."""
    norm = normalize_arabic_text(text)
    # Match words and numbers
    tokens = re.findall(r'[\w\d]+', norm)
    return tokens

class BM25Indexer:
    """
    BM25 Okapi Keyword Search Engine for Arabic RAG (k1=1.5, b=0.75).
    Persists document corpus and inverted index to disk.
    """
    def __init__(self, index_path: Path = BM25_INDEX_PATH, k1: float = 1.5, b: float = 0.75):
        self.index_path = Path(index_path)
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []  # List of chunk objects
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Appends new child chunks and updates BM25 statistics."""
        if not chunks:
            return

        for chunk in chunks:
            tokens = tokenize_arabic(chunk.get("text", ""))
            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            
            self.corpus.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "parent_id": chunk.get("parent_id"),
                "parent_text": chunk.get("parent_text"),
                "metadata": chunk.get("metadata", {})
            })
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            self.doc_freqs.append(freqs)

        self._calc_idf()
        self.save()

    def _calc_idf(self):
        """Calculates IDF scores for all unique tokens in corpus."""
        N = len(self.corpus)
        if N == 0:
            self.avgdl = 0.0
            self.idf = {}
            return

        self.avgdl = sum(self.doc_len) / N
        df: Dict[str, int] = {}
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] = df.get(term, 0) + 1

        self.idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF with smoothing
            idf_val = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(idf_val, 0.01)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches query against BM25 index and returns top_k candidate chunks with BM25 scores.
        """
        if not self.corpus or not query.strip():
            return []

        q_tokens = tokenize_arabic(query)
        if not q_tokens:
            return []

        scores = [0.0] * len(self.corpus)
        N = len(self.corpus)

        for i in range(N):
            d_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            score = 0.0

            for q_term in q_tokens:
                if q_term not in freqs:
                    continue
                f = freqs[q_term]
                idf_val = self.idf.get(q_term, 0.01)
                
                # BM25 Okapi TF formula
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * (d_len / (self.avgdl or 1.0)))
                score += idf_val * (num / den)

            scores[i] = score

        # Combine with corpus chunks
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0.0:  # Only include non-zero keyword matches
                results.append((self.corpus[idx], scores[idx]))

        return results

    def save(self):
        """Saves BM25 index state to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "corpus": self.corpus,
                    "doc_tokens": self.doc_tokens,
                    "doc_len": self.doc_len,
                    "avgdl": self.avgdl,
                    "doc_freqs": self.doc_freqs,
                    "idf": self.idf
                }, f)
        except Exception as e:
            print(f"[BM25 Warning] Failed to save BM25 index: {e}")

    def load(self):
        """Loads BM25 index state from disk."""
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.corpus = data.get("corpus", [])
                self.doc_tokens = data.get("doc_tokens", [])
                self.doc_len = data.get("doc_len", [])
                self.avgdl = data.get("avgdl", 0.0)
                self.doc_freqs = data.get("doc_freqs", [])
                self.idf = data.get("idf", {})
        except Exception as e:
            print(f"[BM25 Warning] Failed to load BM25 index: {e}")

    def reset(self):
        """Resets and clears the BM25 index."""
        self.corpus = []
        self.doc_tokens = []
        self.doc_len = []
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        if self.index_path.exists():
            try:
                self.index_path.unlink()
            except Exception:
                pass
