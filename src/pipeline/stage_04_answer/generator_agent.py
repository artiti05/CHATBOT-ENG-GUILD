import os
import re
import json
import time
import requests
from typing import Dict, Any, List, Tuple, Optional, Generator

from src.config import (
    OLLAMA_URL, OLLAMA_CHAT_MODEL, OLLAMA_KEEP_ALIVE,
    NUM_CTX, NUM_PREDICT, RELEVANCE_THRESHOLD
)


_THINK_OPEN_RE = re.compile(r'<\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r'</\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_PAIR_RE = re.compile(
    r'<\s*(think|thinking|reasoning)\s*>.*?</\s*\1\s*>',
    re.IGNORECASE | re.DOTALL
)


def strip_reasoning(text: str) -> str:
    if not text:
        return ""

    text = _THINK_PAIR_RE.sub('', text)

    m_close = _THINK_CLOSE_RE.search(text)
    if m_close and not _THINK_OPEN_RE.search(text[:m_close.start()]):
        text = text[m_close.end():]

    m_open = _THINK_OPEN_RE.search(text)
    if m_open and not _THINK_CLOSE_RE.search(text[m_open.end():]):
        tail = text[m_open.end():]
        parts = re.split(r'\n\s*\n', tail.strip())
        text = text[:m_open.start()] + (parts[-1] if parts else "")

    return text.strip()


class ReasoningStreamFilter:
    _MAX_TAG_LEN = len('</reasoning>') + 4

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
        if remainder and _THINK_OPEN_RE.match(remainder.strip()):
            return ""
        return remainder


def clean_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = strip_reasoning(text)
    cleaned = re.sub(r'\*+', '', cleaned)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class ResponseGeneratorAgent:
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = OLLAMA_CHAT_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def build_prompt(self,
                     query: str,
                     standalone_query: str,
                     sources: List[Dict[str, Any]],
                     detected_accent: str = "msa") -> Tuple[str, List[Dict[str, Any]]]:
        MAX_CONTEXT_CHARS = 9000
        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        for src in sources:
            score = src.get("similarity_score", 0)
            rank = len(included_sources) + 1
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = src.get("text", "").strip()
            if not raw_text:
                continue

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

        acc_lower = str(detected_accent).lower().strip()
        is_english = acc_lower in ("english", "en", "mixed") or acc_lower.startswith("en")
        if is_english:
            system_prompt = (
                f"You are an Eng-Guild information assistant for the Jordan Engineers Association (JEA).\n"
                f"Strict Response Guidelines (AGENTS.md):\n"
                f"1. GIVE THE DIRECT ANSWER FIRST in sentence 1 at the very top. ZERO preamble, ZERO introductory filler, ZERO greeting.\n"
                f"2. Mandatory Structure: Direct Answer -> Essential Details -> Actionable Next Steps.\n"
                f"3. Short Sentences & Maximum Density: Write using short, concise sentences. Retain maximum factual detail while eliminating fluff.\n"
                f"4. Language Matching: Respond strictly in professional English.\n"
                f"5. Source Citation: Cite source numbers like [Source 1], [Source 2] next to facts.\n"
                f"6. NO Chatbot Artifacts: NO generic sign-offs ('Let me know if you need anything else'), NO conversational filler.\n"
                f"7. Accuracy: Do NOT invent facts. If missing, state: 'I couldn't find that information in the guild resources.'\n"
            )
            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"Knowledge Base Sources:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"User Question: {query}\n"
                f"{'=' * 60}\n"
                f"Direct Answer:"
            )
        else:
            if detected_accent in ("jordanian", "ar-JO"):
                lang_instruction = "مطابقة اللغة والهجة: أجب باللغة العربية بأسلوب نقابي أردني مباشر ومحترف يطابق سؤال المستخدم."
            else:
                lang_instruction = "مطابقة اللغة والهجة: أجب باللغة العربية الفصحى المباشرة."

            system_prompt = (
                f"أنت مساعد معلومات نقابة المهندسين الأردنيين (JEA).\n"
                f"التزم بقواعد الإجابة الصارمة وفق دليل (AGENTS.md):\n\n"
                f"1. قدم الإجابة المباشرة فوراً في السطر الأول. يمنع منعاً باتاً كتابة مقدمات تمهيدية أو جمل ترحيبية أو تمهيد.\n"
                f"2. الهيكل الإجباري للإجابة: [الإجابة المباشرة] ← [التفاصيل الجوهرية والنقاط الأساسية] ← [الخطوات أو الإجراءات العملية التالية].\n"
                f"3. الجمل القصيرة والكثافة العالية: اكتب في جمل قصيرة ومباشرة، وحافظ على جميع التفاصيل والمعلومات الدقيقة مع إلغاء الكلمات الزائدة الحشو.\n"
                f"4. {lang_instruction}\n"
                f"5. التوثيق: اذكر رقم المصدر [المصدر N] بجانب كل معلومة واستشهاد.\n"
                f"6. خلو من المظاهر الزائفة: يُحظر إضافة أي خاتمة روتينية أو جمل ترحيبية أو عرض مساعدة إضافية في النهاية (مثل 'تختص النقابة...' أو 'في حال وجود استفسارات').\n"
                f"7. الدقة والاعتماد على المصادر: اجب اعتماداً على المعلومات الواردة في المصادر المرفقة بأعلاه، والخص كافة التفاصيل والخطوات والشروط المذكورة بدقة.\n"
            )

            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"المصادر من قاعدة المعرفة:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"سؤال المستخدم: {query}\n"
                f"{'=' * 60}\n"
                f"الإجابة المباشرة:"
            )

        return full_prompt, included_sources

    def generate(self, prompt: str) -> Tuple[str, bool]:
        if not prompt:
            return "", False

        models_to_try = [self.model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
        for model_tag in models_to_try:
            try:
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
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
        if not prompt:
            return

        models_to_try = [self.model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
        for model_tag in models_to_try:
            try:
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
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
