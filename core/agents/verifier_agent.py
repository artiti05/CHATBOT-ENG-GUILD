import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional


def normalize_arabic_stem(text: str) -> str:
    """Strips diacritics, tatweel, and normalizes Arabic characters for n-gram stem matching."""
    if not text:
        return ""
    # Strip diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Strip tatweel (kashida)
    text = re.sub(r'\u0640', '', text)
    # Normalize Alef variants -> ا
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Yaa / Alef Maqsura -> ي
    text = re.sub(r'ى', 'ي', text)
    # Normalize Teh Marbuta -> ه
    text = re.sub(r'ة', 'ه', text)
    # Strip common prefixes (الـ)
    text = re.sub(r'\bال', '', text)
    return text.lower().strip()


class VerifierAgent:
    """
    Corrective RAG (CRAG) & Groundedness Verification Agent.
    1. Evaluates retrieval confidence & triggers query expansion re-retrieval loops.
    2. Verifies Arabic n-gram stem groundedness between generated answer and source context blocks.
    3. Enforces strict source citation guardrails ([المصدر N]).
    """

    def __init__(self, confidence_threshold: float = 55.0):
        self.confidence_threshold = confidence_threshold

    def evaluate_retrieval_confidence(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates candidate sources to determine if Corrective RAG (CRAG) re-retrieval is needed.
        """
        if not sources:
            return {"confidence_low": True, "top_score": 0.0, "reason": "no_sources"}

        top_score = sources[0].get("similarity_score", 0.0)
        avg_top3 = sum(s.get("similarity_score", 0.0) for s in sources[:3]) / min(len(sources), 3)

        confidence_low = (top_score < self.confidence_threshold) or (avg_top3 < 52.0)

        return {
            "confidence_low": confidence_low,
            "top_score": top_score,
            "avg_top3": avg_top3,
            "source_count": len(sources)
        }

    def generate_query_expansions(self, query: str) -> List[str]:
        """
        Generates alternative search term expansions for Corrective RAG fallback search.
        """
        clean_q = re.sub(r'[^\w\s]', ' ', query).strip()
        words = clean_q.split()

        expansions = []
        # 1. Main noun phrase (drop common question words)
        stop_question_words = {"ما", "هي", "هو", "كيف", "شروط", "رسوم", "طريقة", "كيفية", "أين", "كم", "ماذا"}
        core_words = [w for w in words if w not in stop_question_words and len(w) > 2]
        if core_words:
            expansions.append(" ".join(core_words))

        # 2. Relaxed stem expansion
        stemmed_words = [normalize_arabic_stem(w) for w in words if len(w) > 2]
        if stemmed_words:
            expansions.append(" ".join(stemmed_words[:4]))

        # Ensure uniqueness while maintaining order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp and exp not in seen and exp != query:
                seen.add(exp)
                unique_expansions.append(exp)

        return unique_expansions

    def verify_answer_groundedness(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Checks generated answer against source context blocks using Arabic stem n-gram overlap.
        Ensures claims are grounded and citation tags [المصدر N] are present.
        """
        if not answer or not answer.strip() or not sources:
            return {"is_grounded": False, "score": 0.0, "cleaned_answer": answer, "warnings": ["empty_input"]}

        # Combine all source text into a single normalized reference corpus
        raw_corpus = " ".join([s.get("text", "") + " " + s.get("parent_text", "") for s in sources])
        norm_corpus = normalize_arabic_stem(raw_corpus)
        corpus_words = set(norm_corpus.split())

        # Split answer into individual sentences / clauses
        sentences = [s.strip() for s in re.split(r'[\.\!\?\n]+', answer) if len(s.strip()) > 10]
        if not sentences:
            return {"is_grounded": True, "score": 100.0, "cleaned_answer": answer, "warnings": []}

        grounded_sentences = 0
        unsupported_sentences = []

        for sent in sentences:
            norm_sent = normalize_arabic_stem(sent)
            sent_words = [w for w in norm_sent.split() if len(w) > 2]
            if not sent_words:
                grounded_sentences += 1
                continue

            # Calculate word overlap with context corpus
            matches = sum(1 for w in sent_words if w in corpus_words)
            overlap_ratio = matches / len(sent_words)

            if overlap_ratio >= 0.35:  # At least 35% stem overlap with retrieved context
                grounded_sentences += 1
            else:
                unsupported_sentences.append(sent)

        groundedness_score = round((grounded_sentences / len(sentences)) * 100, 1)

        # Check citation presence
        has_citations = bool(re.search(r'\[المصدر\s*\d+\]', answer))
        warnings = []
        if not has_citations:
            warnings.append("missing_citation_tags")
        if groundedness_score < 60.0:
            warnings.append("low_groundedness_score")

        return {
            "is_grounded": groundedness_score >= 60.0,
            "score": groundedness_score,
            "has_citations": has_citations,
            "total_sentences": len(sentences),
            "grounded_sentences": grounded_sentences,
            "unsupported_sentences": unsupported_sentences,
            "warnings": warnings,
            "cleaned_answer": answer.strip()
        }
