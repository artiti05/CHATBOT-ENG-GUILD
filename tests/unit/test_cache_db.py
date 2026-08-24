
import pytest

from src.cache_db.document_registry import DocumentRegistry
from src.cache_db.semantic_cache import SemanticCache


@pytest.mark.unit
class TestSemanticCache:
    def test_cache_put_and_get(self, tmp_path):
        cache_file = tmp_path / "test_cache.json"
        cache = SemanticCache(persist_path=cache_file)

        query = "ما هي شروط تسجيل المهندسين الأردنيين؟"
        answer = "شروط التسجيل: البكالوريوس وكشف العلامات والهوية ودفع 136 دينار."
        sources = [{"title": "شروط التسجيل", "similarity_score": 95.0}]

        # Initially empty
        assert cache.get(query) is None

        # Put entry
        cache.put(query, answer, sources)

        # Retrieve entry
        hit = cache.get(query)
        assert hit is not None
        assert hit["answer"] == answer
        assert len(hit["sources"]) == 1

    def test_cache_normalization(self, tmp_path):
        cache_file = tmp_path / "test_cache.json"
        cache = SemanticCache(persist_path=cache_file)

        cache.put("  SALARY SCALE  ", "Answer", [])
        assert cache.get("salary scale") is not None

    def test_cache_clear(self, tmp_path):
        cache_file = tmp_path / "test_cache.json"
        cache = SemanticCache(persist_path=cache_file)

        cache.put("Q1", "A1", [])
        assert len(cache.entries) == 1
        cache.clear()
        assert len(cache.entries) == 0


@pytest.mark.unit
class TestDocumentRegistry:
    def test_registry_init_and_operations(self, tmp_path):
        db_file = tmp_path / "test_registry.db"
        registry = DocumentRegistry(db_path=db_file)

        doc_dict = {
            "source_id": "test_doc_1",
            "file_name": "test_file.pdf",
            "source_type": "PDF",
            "content_hash": "abc123hash",
            "total_pages": 5,
            "parsed_md_path": "/tmp/parsed.md"
        }

        sample_path = tmp_path / "test_file.pdf"
        sample_path.touch()

        # Register document
        registry.register_document(doc_dict, sample_path, status="active")

        # Get document
        retrieved = registry.get_document("test_doc_1")
        assert retrieved is not None
        assert retrieved["file_name"] == "test_file.pdf"
        assert retrieved["status"] == "active"

        # List documents
        docs = registry.list_documents()
        assert len(docs) >= 1

        # Update status
        registry.update_status("test_doc_1", "excluded")
        updated = registry.get_document("test_doc_1")
        assert updated["status"] == "excluded"
