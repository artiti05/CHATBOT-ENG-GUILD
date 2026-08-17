import time
from typing import Dict, Any, List, Optional

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
            "language_detection": 0.0,
            "normalization": 0.0,
            "msa_conversion": 0.0,
            "intent": 0.0,
            "bm25": 0.0,
            "bge_m3": 0.0,
            "rapidfuzz": 0.0,
            "rrf": 0.0,
            "reranker": 0.0,
            "priority": 0.0,
            "crag": 0.0,
            "answer": 0.0
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
            f"TOTAL REQUEST",
            f"Total latency: {int(total)} ms",
            "",
            "Breakdown:"
        ]
        labels = {
            "cache": "Cache",
            "language_detection": "Language detection",
            "normalization": "Normalization",
            "msa_conversion": "MSA conversion",
            "intent": "Intent",
            "bm25": "BM25",
            "bge_m3": "BGE-M3",
            "rapidfuzz": "RapidFuzz",
            "rrf": "RRF",
            "reranker": "Reranker",
            "priority": "Priority",
            "crag": "CRAG",
            "answer": "Answer"
        }
        for key, label in labels.items():
            val = int(self.stage_latencies.get(key, 0.0))
            lines.append(f"{label:<20} {val:>3} ms")
        return "\n".join(lines)
