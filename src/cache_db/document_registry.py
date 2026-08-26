import os
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import REGISTRY_DB_PATH, STORAGE_DIR, CRAWL_CACHE_DIR

class DocumentRegistry:
    def __init__(self, db_path: Path = REGISTRY_DB_PATH):
        self.db_path = Path(db_path).resolve()
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_documents (
                    source_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    total_pages INTEGER DEFAULT 1,
                    parsed_md_path TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            try:
                cursor.execute("ALTER TABLE processed_documents ADD COLUMN parsed_md_path TEXT DEFAULT '';")
            except Exception:
                pass
            conn.commit()

    def register_document(self, doc_dict: Dict[str, Any], file_path: Path, status: str = "active"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processed_documents 
                (source_id, file_name, file_path, source_type, content_hash, total_pages, parsed_md_path, status, last_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
            """, (
                doc_dict["source_id"],
                doc_dict["file_name"],
                str(file_path.resolve()),
                doc_dict["source_type"],
                doc_dict.get("content_hash", ""),
                doc_dict.get("total_pages", 1),
                doc_dict.get("parsed_md_path", ""),
                status
            ))
            conn.commit()

    def get_document(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processed_documents WHERE source_id = ?;", (source_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processed_documents ORDER BY last_processed DESC;")
            return [dict(r) for r in cursor.fetchall()]

    def update_status(self, source_id: str, status: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE processed_documents SET status = ? WHERE source_id = ?;", (status, source_id))
            conn.commit()

class AdminManager:
    def __init__(self):
        self.registry = DocumentRegistry()
        self.pipeline = IngestionPipeline()

    def process_and_index_document(self, file_path: Path, status: str = "active") -> Dict[str, Any]:
        file_path = Path(file_path).resolve()
        res = self.pipeline.process_file(file_path)
        doc = res["document"]
        self.registry.register_document(doc, file_path, status=status)
        return res

    def exclude_document(self, source_id: str):
        self.pipeline.indexer.delete_document_chunks(source_id)
        self.registry.update_status(source_id, "excluded")

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.registry.list_documents()

class TicketRegistry:
    """Persists AI-triggered support tickets (see
    src/pipeline/stage_05_ticket/ticket_agent.py). Ticket resolution itself is
    a purely human process -- this class only stores and lists what employees
    need to act on."""

    def __init__(self, db_path: Path = REGISTRY_DB_PATH):
        self.db_path = Path(db_path).resolve()
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT NOT NULL,
                    issue_text TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    phone TEXT DEFAULT '',
                    engineer_number TEXT DEFAULT '',
                    raw_contact_text TEXT DEFAULT '',
                    status TEXT DEFAULT 'open'
                );
            """)
            for ddl in (
                "ALTER TABLE tickets ADD COLUMN title TEXT DEFAULT '';",
                "ALTER TABLE tickets ADD COLUMN priority TEXT DEFAULT 'medium';",
                "ALTER TABLE tickets ADD COLUMN conversation_context TEXT DEFAULT '';",
                "ALTER TABLE tickets ADD COLUMN user_id TEXT DEFAULT '';",
            ):
                try:
                    cursor.execute(ddl)
                except Exception:
                    pass
            conn.commit()

    def create_ticket(
        self,
        reason: str,
        query: str,
        history: List[Dict[str, str]],
        user: Dict[str, str],
        priority: str = "medium",
    ) -> int:
        title = query[:40] + ("..." if len(query) > 40 else "")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tickets
                    (reason, issue_text, name, phone, engineer_number, status, title, priority, conversation_context, user_id)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?);
            """, (
                reason,
                query,
                user.get("name", ""),
                user.get("phone", ""),
                user.get("engineer_number", ""),
                title,
                priority,
                json.dumps(history, ensure_ascii=False),
                user.get("user_id", ""),
            ))
            conn.commit()
            return cursor.lastrowid

    def list_tickets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC;", (status,))
            else:
                cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC;")
            return [dict(r) for r in cursor.fetchall()]

    def get_ticket(self, ticket_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tickets WHERE id = ?;", (ticket_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_status(self, ticket_id: int, status: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE tickets SET status = ? WHERE id = ?;", (status, ticket_id))
            conn.commit()


class WebsiteCrawlerCache:
    def __init__(self, cache_dir: Path = CRAWL_CACHE_DIR):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_crawled_page(self, url: str, content: str) -> Path:
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
        file_path = self.cache_dir / f"crawl_{url_hash}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Source URL: {url}\n\n{content}")
        return file_path
