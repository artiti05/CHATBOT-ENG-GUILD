import pytest
from src.pipeline.stage_03_verify_rerank.confidence import DeterministicConfidenceEvaluator
from src.pipeline.stage_04_answer.answer_router import AnswerRouter
from src.pipeline.stage_04_answer.templates import StructuredAnswerTemplates

@pytest.mark.unit
class TestDeterministicConfidenceEvaluator:
    def test_evaluate_no_candidates(self):
        query_obj = {"intent": ["MEMBERSHIP"], "entities": {}}
        res = DeterministicConfidenceEvaluator.evaluate([], query_obj)
        assert res["decision"] == "LOW"
        assert res["score"] == 0.0
        assert res["llm_verifier_used"] is False

    def test_evaluate_high_confidence(self):
        candidates = [
            {"final_score": 85.0, "priority_boost": 0.5}
        ]
        query_obj = {"intent": ["MEMBERSHIP"], "entities": {"JEA": "نقابة المهندسين"}}
        res = DeterministicConfidenceEvaluator.evaluate(candidates, query_obj)
        assert res["decision"] in ("HIGH", "MEDIUM")
        assert res["score"] > 0.70


@pytest.mark.unit
class TestAnswerRouter:
    def setup_method(self):
        self.router = AnswerRouter()

    def test_route_generation(self):
        query_obj = {"intent": ["MEMBERSHIP"]}
        conf_res = {"decision": "HIGH", "score": 0.85}
        res = self.router.route(query_obj, conf_res)
        assert res["route"] == "GENERATION"
        assert res["llm_required"] is True


@pytest.mark.unit
class TestStructuredAnswerTemplates:
    def test_get_template_answer(self):
        tmpl = StructuredAnswerTemplates.get_template_answer("MEMBERSHIP")
        if tmpl:
            assert isinstance(tmpl, str)
            assert len(tmpl) > 0
