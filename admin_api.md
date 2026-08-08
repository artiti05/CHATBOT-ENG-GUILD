# Admin API & Panel — Knowledge Base Management

Manage the documents in the vector knowledge base at runtime — no manual
`main.py ingest` runs needed.

## Auth

All admin endpoints use the standard `X-API-Key` header, validated against
`ADMIN_API_KEY` (separate from `USER_API_KEY` used by the chat UI — a leaked
chat key can never mutate the knowledge base). Wrong/missing key → 403.

Generate keys with:
```
python -c "import secrets; print(secrets.token_hex(32))"
```

## Browser panel

Open `http://<host>:8080/admin` — enter the admin key once (stored in the
browser's localStorage). The panel supports listing, uploading (PDF/TXT/MD),
deleting, and re-ingesting documents.

## Endpoints (base: `/api/admin`)

### GET /files
List indexed documents.
Response: `{"documents": [{"source_id": ..., "status": ..., ...}], "count": N}`

### POST /files/upload  → 202
Multipart upload (`file` field, .pdf/.txt/.md only). The file is saved and
processed in a **background task** — the 202 response returns immediately:
`{"status": "processing_started", "filename": ...}`. Poll `GET /files` to see
when the document appears.

### DELETE /files/{filename}
Removes the document from the registry and its vectors from ChromaDB.
404 if the filename is not in the knowledge base.

### POST /files/re-ingest  → 202
Body: `{"files": ["name1.pdf"], "reingest_all": false}`
Re-processes the named files (or everything with `reingest_all: true`) from
the data directories. Existing entries are excluded first to avoid duplicate
vectors.

## Operational notes

1. Upload/re-ingest run in FastAPI background tasks **inside the serving
   process**. Heavy PDF parsing (vision model) will compete with query
   serving for CPU/RAM — schedule large batch ingestions during quiet hours.
2. Ingestion and serving share one ChromaDB `PersistentClient`, which is why
   runtime ingestion happens in-process rather than as a separate script
   against live storage (Chroma does not support two writer processes).
3. Embeddings are BGE-M3 (hardcoded in `core/ingestion.BGEM3Embedder`);
   index and queries always use the same model.