from typing import Dict, Any, List

class DiagnosticAnalyzer:
    """Categorizes pipeline bottlenecks and operational flags automatically."""

    @staticmethod
    def analyze_trace(report: Dict[str, Any]) -> List[str]:
        diagnostics = []
        breakdown = report.get("breakdown", {})
        metadata = report.get("metadata", {})

        bge_latency = breakdown.get("bge_m3", 0.0)
        reranker_latency = breakdown.get("reranker", 0.0)
        if bge_latency > 150.0:
            diagnostics.append(f"Slow Component: BGE-M3 latency high ({bge_latency} ms)")
        if reranker_latency > 200.0:
            diagnostics.append(f"Slow Component: Reranker latency high ({reranker_latency} ms)")

        crag_meta = metadata.get("crag", {})
        if crag_meta.get("llm_used", False):
            diagnostics.append("CRAG Overuse: LLM Verifier invoked")

        ans_meta = metadata.get("answer", {})
        if ans_meta.get("llm_used", False):
            diagnostics.append("LLM Generator Used: Route set to GENERATION")
        else:
            diagnostics.append("Deterministic Route Used: LLM bypassed")

        return diagnostics
