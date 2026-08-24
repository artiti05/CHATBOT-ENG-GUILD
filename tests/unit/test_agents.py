import pytest

from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent, clean_formatting, strip_reasoning


@pytest.mark.unit
class TestResponseGeneratorAgent:
    def setup_method(self):
        self.agent = ResponseGeneratorAgent()

    def test_strip_reasoning_well_formed(self):
        text = "<think>Some inner model reasoning</think>Direct answer text"
        assert strip_reasoning(text) == "Direct answer text"

    def test_strip_reasoning_orphan_close(self):
        text = "internal model trace</think>\nActual response"
        assert strip_reasoning(text) == "Actual response"

    def test_clean_formatting(self):
        raw = "### Header\n**Bold Text** and *italic*"
        cleaned = clean_formatting(raw)
        assert "#" not in cleaned
        assert "*" not in cleaned
        assert "Bold Text" in cleaned

    def test_build_prompt_empty_sources(self):
        prompt, sources = self.agent.build_prompt("Query", "Query", [], "msa")
        assert prompt == ""
        assert sources == []

    def test_build_prompt_with_valid_sources(self, sample_documents):
        prompt, included = self.agent.build_prompt(
            query="ما هي شروط التقديم؟",
            standalone_query="ما هي شروط التقديم؟",
            sources=sample_documents,
            detected_accent="msa"
        )
        assert len(prompt) > 0
        assert len(included) > 0
        assert "شروط-تسجيل-الاردنيين.md" in prompt or "شروط تسجيل المهندسين الأردنيين" in prompt
