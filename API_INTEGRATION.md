# RAG Service — Integration Guide

Base URL: http://<host>:8080
Auth: header `X-API-Key: <key>` on all POST endpoints.

## POST /api/chat
Request:
{
  "query": "شو الأوراق المطلوبة للانتساب؟",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "top_k": 10
}

Response:
{
  "query": "...",
  "normalized_query": "...",
  "detected_accent": "jordanian | msa | english",
  "answer": "...",
  "sources": [ {"rank": 1, "title": "...", "text": "...", "similarity_score": 87, ...} ],
  "llm_connected": true,
  "time_taken": 8.4
}

## Integration rules
1. The service is STATELESS. The caller owns conversation history and
   sends the recent turns each call (the engine uses the last 6).
2. Always check `llm_connected`. If false, Ollama was unreachable and
   `answer` is a canned fallback — decide whether to show it or retry.
3. Set your HTTP client timeout to 90s. Typical latency is 5–20s
   (reranking + local 7B generation).
4. GET /api/health returns {"status":"ok","chunks":N} — use for
   readiness probes. Note: first boot takes minutes (model downloads).

## Ingestion (offline, separate from serving)
Run `python main.py ingest` on the GPU machine. The serving container
only needs the resulting `storage/` directory. Never run ingestion
against a live serving container's storage — ChromaDB does not support
two writer processes.
## Known notes (verified 2026-08-04)
1. Index built with EMBEDDING_PROVIDER=bge-m3. The same provider must be
   set at query time; changing it requires full re-ingestion.
2. Query in Arabic for best results — the reranker scores cross-lingual
   (English) queries near zero against the Arabic corpus.
3. Answer relevance with qwen2.5:7b is inconsistent (correct sources are
   retrieved but the model can drift off-topic). Use top_k=5, and plan a
   stronger generation model for production.