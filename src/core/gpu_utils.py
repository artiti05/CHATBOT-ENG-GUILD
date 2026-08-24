"""Shared GPU VRAM availability check for locally-hosted models
(embedder, reranker, AraT5 rewriter).

Checks free VRAM at model-load time so these models only claim the GPU when
there's enough room left after the vLLM server's allocation.
"""


def pick_device(min_free_gb: float = 2.0) -> str:
    """Returns 'cuda' only if a GPU is present with at least min_free_gb free VRAM, else 'cpu'."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        if free_gb >= min_free_gb:
            return "cuda"
        print(f"[GPU] Only {free_gb:.2f}GB VRAM free (< {min_free_gb}GB required) — falling back to CPU.")
        return "cpu"
    except Exception as e:
        print(f"[GPU] VRAM check failed ({e}) — falling back to CPU.")
        return "cpu"
