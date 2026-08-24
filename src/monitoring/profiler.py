import time
from typing import Any, Dict, Optional


class StageTimer:
    """Context manager for timing pipeline stage execution."""
    def __init__(self, profiler: 'PipelineProfiler', stage_name: str):
        self.profiler = profiler
        self.stage_name = stage_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.profiler.record_stage(self.stage_name, elapsed_ms)


class PipelineProfiler:
    """First-class profiling component measuring 13-stage pipeline latency & diagnostic metadata."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.perf_counter()
        self.stage_latencies: Dict[str, float] = {
            "cache": 0.0,
            "normalization": 0.0,
            "dialect_rewrite": 0.0,
            "dense_search": 0.0,
            "sparse_search": 0.0,
            "fusion": 0.0,
            "priority_boost": 0.0,
            "rerank": 0.0,
            "generation": 0.0
        }
        self.stage_metadata: Dict[str, Any] = {}

    def time_stage(self, stage_name: str) -> StageTimer:
        return StageTimer(self, stage_name)

    def record_stage(self, stage_name: str, latency_ms: float, metadata: Optional[Dict[str, Any]] = None):
        self.stage_latencies[stage_name] = round(latency_ms, 2)
        if metadata:
            self.stage_metadata[stage_name] = metadata

    def get_total_latency(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)

    def generate_report(self) -> Dict[str, Any]:
        total = self.get_total_latency()
        return {
            "request_id": self.request_id,
            "total_latency_ms": total,
            "breakdown": self.stage_latencies,
            "metadata": self.stage_metadata
        }

    def format_text_breakdown(self) -> str:
        total = self.get_total_latency()
        lines = [
            f"REQUEST {self.request_id}",
            "TOTAL REQUEST",
            f"Total latency: {int(total)} ms",
            "",
            "Breakdown:"
        ]
        labels = {
            "cache": "Cache Lookup",
            "normalization": "Query Normalization",
            "dialect_rewrite": "AraT5 MSA Rewrite",
            "dense_search": "Chroma Dense Search",
            "sparse_search": "BM25 Sparse Search",
            "fusion": "RRF Fusion",
            "priority_boost": "Doc Priority Boost",
            "rerank": "Cross-Encoder Rerank",
            "generation": "LLM Generation"
        }
        for key, label in labels.items():
            val = int(self.stage_latencies.get(key, 0.0))
            lines.append(f"{label:<25} {val:>3} ms")
        return "\n".join(lines)


    def write_request_log(self, user_query: str, answer: str, cache_hit: bool = False, route: str = "GENERATION"):
        """Appends structured JSON log entry using a 10MB RotatingFileHandler to prevent disk growth."""
        try:
            import json
            import logging
            from logging.handlers import RotatingFileHandler

            from src.config import LOGS_DIR

            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / "chat_requests.log"

            logger = logging.getLogger("chat_requests")
            if not logger.handlers:
                handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": self.request_id,
                "query": user_query,
                "answer_snippet": answer[:200] if answer else "",
                "cache_hit": cache_hit,
                "route": route,
                "total_latency_ms": self.get_total_latency(),
                "breakdown": self.stage_latencies
            }

            logger.info(json.dumps(log_entry, ensure_ascii=False))
        except Exception as e:
            print(f"Warning: Failed to write request log: {e}")
