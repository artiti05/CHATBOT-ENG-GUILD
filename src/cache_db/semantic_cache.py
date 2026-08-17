import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_ENTRIES, CACHE_PERSIST_PATH

class SemanticCache:
    """Fast semantic cache intercepting queries for < 10ms responses."""

    def __init__(self, persist_path: Path = CACHE_PERSIST_PATH):
        self.persist_path = Path(persist_path)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        q_norm = query.strip().lower()
        for entry in self.entries:
            if entry.get("query_normalized") == q_norm:
                return entry
        return None

    def put(self, query: str, answer: str, sources: List[Dict[str, Any]]):
        q_norm = query.strip().lower()
        entry = {
            "query_normalized": q_norm,
            "answer": answer,
            "sources": sources
        }
        self.entries.insert(0, entry)
        self.entries = self.entries[:CACHE_MAX_ENTRIES]
        self._save()

    def clear(self):
        self.entries = []
        self._save()

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
