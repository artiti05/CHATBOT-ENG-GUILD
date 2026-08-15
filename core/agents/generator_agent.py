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


# ---------------------------------------------------------------------------
# REASONING / THINKING TRACE SUPPRESSION
# Thinking-capable models (qwen3, deepseek-r1, magistral, ministral reasoning
# builds) emit <think>...</think> traces inside the response. These must never
# reach the end-user UI.
# ---------------------------------------------------------------------------
_THINK_OPEN_RE = re.compile(r'<\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r'</\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_PAIR_RE = re.compile(
    r'<\s*(think|thinking|reasoning)\s*>.*?</\s*\1\s*>',
    re.IGNORECASE | re.DOTALL
)


def strip_reasoning(text: str) -> str:
    """Removes model reasoning traces from a complete (batch) response.

    Handles three failure shapes:
    1. Well-formed <think>...</think> blocks anywhere in the text.
    2. Orphan closing tag (chat template auto-opened the think block, so the
       raw output starts mid-reasoning and only contains </think>): everything
       up to and including the first orphan close tag is dropped.
    3. Unclosed opening tag at the very start (model never closed it but a
       blank-line separated answer follows): the tag itself is removed and the
       last paragraph is kept as the answer.
    """
    if not text:
        return ""

    # Case 1: well-formed pairs
    text = _THINK_PAIR_RE.sub('', text)

    # Case 2: orphan close tag with no preceding open tag
    m_close = _THINK_CLOSE_RE.search(text)
    if m_close and not _THINK_OPEN_RE.search(text[:m_close.start()]):
        text = text[m_close.end():]

    # Case 3: unclosed open tag — keep content after the last blank line
    m_open = _THINK_OPEN_RE.search(text)
    if m_open and not _THINK_CLOSE_RE.search(text[m_open.end():]):
        tail = text[m_open.end():]
        parts = re.split(r'\n\s*\n', tail.strip())
        text = text[:m_open.start()] + (parts[-1] if parts else "")

    return text.strip()


class ReasoningStreamFilter:
    """Stateful token filter for streaming: suppresses everything between
    <think>/<thinking>/<reasoning> open and close tags, even when tags are
    split across token boundaries. Feed each token; emit whatever it returns;
    call flush() after the stream ends."""

    _MAX_TAG_LEN = len('</reasoning>') + 4  # partial-tag holdback window

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        self._buf += token
        out = []
        while True:
            if self._in_think:
                m = _THINK_CLOSE_RE.search(self._buf)
                if m:
                    self._buf = self._buf[m.end():]
                    self._in_think = False
                    continue
                # hold back only the tail that could be a partial closing tag
                self._buf = self._buf[-self._MAX_TAG_LEN:]
                break
            else:
                m = _THINK_OPEN_RE.search(self._buf)
                if m:
                    out.append(self._buf[:m.start()])
                    self._buf = self._buf[m.end():]
                    self._in_think = True
                    continue
                safe_len = len(self._buf) - self._MAX_TAG_LEN
                if safe_len > 0:
                    out.append(self._buf[:safe_len])
                    self._buf = self._buf[safe_len:]
                break
        return "".join(out)

    def flush(self) -> str:
        remainder = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        # a lone partial tag left in the buffer is not reasoning — emit it,
        # but drop it if it is purely an unfinished tag fragment
        if remainder and _THINK_OPEN_RE.match(remainder.strip()):
            return ""
        return remainder


def clean_formatting(text: str) -> str:
    """Strips reasoning traces, then markdown bold/italic asterisks (**text**)
    and header hashes for clean text delivery."""
    if not text:
        return ""
    cleaned = strip_reasoning(text)
    cleaned = re.sub(r'\*+', '', cleaned)
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
                    "think": False,  # disable native reasoning traces (Ollama >= 0.9; ignored otherwise)
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
                    "think": False,  # disable native reasoning traces (Ollama >= 0.9; ignored otherwise)
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": NUM_PREDICT,
                        "num_ctx": NUM_CTX
                    }
                }
                res = requests.post(self.ollama_url, json=payload, stream=True, timeout=(3.0, 120.0))
                if res.status_code == 200:
                    reasoning_filter = ReasoningStreamFilter()
                    for line in res.iter_lines():
                        if line:
                            try:
                                chunk_json = json.loads(line.decode('utf-8'))
                                token = chunk_json.get("response", "")
                                # some Ollama builds emit thinking in a separate
                                # field — never forward it
                                safe = reasoning_filter.feed(token)
                                safe = safe.replace("*", "").replace("#", "")
                                if safe:
                                    yield safe
                            except Exception:
                                pass
                    tail = reasoning_filter.flush()
                    tail = tail.replace("*", "").replace("#", "")
                    if tail:
                        yield tail
                    return
            except Exception:
                continue
