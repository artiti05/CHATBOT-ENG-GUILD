import os
import re
import json
import time
import requests
from typing import Dict, Any, List, Tuple, Optional, Generator

from core.config import (
    OLLAMA_URL, OLLAMA_CHAT_MODEL, OLLAMA_KEEP_ALIVE,
    NUM_CTX, NUM_PREDICT, RELEVANCE_THRESHOLD
)


def clean_formatting(text: str) -> str:
    """Strips markdown bold/italic asterisks (**text**) and header hashes for clean text delivery."""
    if not text:
        return ""
    cleaned = re.sub(r'\*+', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class ResponseGeneratorAgent:
    """
    Specialized Agentic Response Generator.
    Responsible for context formatting, persona tone instruction,
    and streaming/batch LLM generation calls to Ollama.
    """

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = OLLAMA_CHAT_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def build_prompt(self,
                     query: str,
                     standalone_query: str,
                     sources: List[Dict[str, Any]],
                     detected_accent: str = "msa") -> Tuple[str, List[Dict[str, Any]]]:
        """Formats source documents and constructs system & user generation prompt."""
        MAX_CONTEXT_CHARS = 9000
        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        for src in sources:
            score = src.get("similarity_score", 0)
            if score < RELEVANCE_THRESHOLD:
                continue

            rank = src.get("rank", len(included_sources) + 1)
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = src.get("text", "").strip()

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (نسبة التطابق: {score}%)"

            block_str = f"{header}\n{raw_text}\n"
            if accumulated_chars + len(block_str) > MAX_CONTEXT_CHARS and context_blocks:
                break

            context_blocks.append(block_str)
            accumulated_chars += len(block_str)
            included_sources.append(src)

        if not included_sources:
            return "", []

        context_str = "\n" + ("=" * 50) + "\n" + "\n".join(context_blocks) + ("=" * 50)

        num_sources = len(included_sources)

        if detected_accent == "english":
            system_prompt = (
                f"You are an AI Assistant specializing in the Jordan Engineers Association (JEA).\n"
                f"Your task: Read the provided Knowledge Base sources carefully and synthesize a clear, comprehensive answer.\n"
                f"You have {num_sources} knowledge base sources for this question.\n\n"
                f"CRITICAL MANDATORY REQUIREMENT: YOUR ENTIRE RESPONSE MUST BE WRITTEN IN CLEAR, PROFESSIONAL ENGLISH.\n"
                f"Translate all relevant information from the Arabic sources into fluent English.\n"
                f"Cite source numbers like [Source 1], [Source 2] next to facts.\n"
                f"Do NOT invent facts. Do NOT use markdown symbols (* or ** or ##). Use plain numbers and text bullet points only.\n"
            )
            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"Knowledge Base Sources:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"User Question: {query}\n"
                f"{'=' * 60}\n"
                f"Answer in English:"
            )
        else:
            if detected_accent == "jordanian":
                lang_instruction = "لغة الإجابة: اللهجة الأردنية الودية والواضحة — خاطب المهندس بأسلوب نقابي حميمي ومتعاطف."
            else:
                lang_instruction = "لغة الإجابة: العربية الفصحى الرسمية — أسلوب واضح ومحترف يليق بنقابة مهنية."

            system_prompt = (
                f"أنت مساعد ذكي متخصص في شؤون نقابة المهندسين الأردنيين (Jordan Engineers Association — JEA).\n"
                f"مهمتك: قراءة المصادر المرفقة بعناية، استيعابها، والإجابة بطريقة تركيبية منطقية.\n"
                f"لديك {num_sources} مصدر من قاعدة المعرفة لهذا السؤال.\n\n"
                f"أسلوب الإجابة المطلوب:\n"
                f"• ابدأ بجملة تمهيدية واحدة موجزة تحدد محور الإجابة.\n"
                f"• اقرأ جميع المصادر واستخلص كل المعلومات ذات الصلة بالسؤال.\n"
                f"• اربط المعلومات من مصادر مختلفة، واذكر رقم المصدر [المصدر N] عند كل معلومة.\n"
                f"• إذا كانت صياغة المصدر تقنية، وضّحها بلغة بسيطة دون إضافة معلومات خارجية.\n"
                f"• اختم بجملة تلخص الفكرة الرئيسية أو تظهر الترحيب بالاستفسارات.\n\n"
                f"قواعد صارمة:\n"
                f"1. لا تخترع أي معلومة. كل فكرة يجب أن تكون مستندة إلى المصادر المرفقة.\n"
                f"2. إذا لم تجد إجابة في المصادر، قل ذلك صراحةً بدلاً من التكهن.\n"
                f"3. يُحظر استخدام رموز Markdown (* أو ** أو ##). استخدم الأرقام والنقاط النصية فقط.\n"
                f"4. {lang_instruction}\n"
            )

            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"المصادر من قاعدة المعرفة:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"سؤال المستخدم: {standalone_query}\n"
                f"{'=' * 60}\n"
                f"الإجابة:"
            )

        return full_prompt, included_sources


    def generate(self, prompt: str) -> Tuple[str, bool]:
        """Executes synchronous batch generation with model fallback."""
        if not prompt:
            return "", False

        models_to_try = [self.model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
        for model_tag in models_to_try:
            try:
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": NUM_PREDICT,
                        "num_ctx": NUM_CTX
                    }
                }
                res = requests.post(self.ollama_url, json=payload, timeout=(3.0, 120.0))
                if res.status_code == 200:
                    content = res.json().get("response", "").strip()
                    if content:
                        return clean_formatting(content), True
            except Exception:
                continue

        return "", False

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Yields real-time token chunks directly from Ollama stream=True."""
        if not prompt:
            return

        models_to_try = [self.model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
        for model_tag in models_to_try:
            try:
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": NUM_PREDICT,
                        "num_ctx": NUM_CTX
                    }
                }
                res = requests.post(self.ollama_url, json=payload, stream=True, timeout=(3.0, 120.0))
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            try:
                                chunk_json = json.loads(line.decode('utf-8'))
                                token = chunk_json.get("response", "")
                                token = token.replace("*", "").replace("#", "")
                                if token:
                                    yield token
                            except Exception:
                                pass
                    return
            except Exception:
                continue
