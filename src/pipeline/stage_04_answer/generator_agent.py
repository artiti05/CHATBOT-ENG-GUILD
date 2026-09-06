import re
from typing import Any, Dict, Generator, List, Tuple, Union

import openai

from src.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
)

# ---------------------------------------------------------------------------
# Reasoning-tag stripping helpers (unchanged — kept for model compatibility)
# ---------------------------------------------------------------------------
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
    # Strip any inline source tags like [المصدر 1] or [Source 2]
    cleaned = re.sub(r'\[\s*(المصدر|المصادر|Source|Sources)\s*[\d\.,\s]+\]', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# ResponseGeneratorAgent
# ---------------------------------------------------------------------------

class ResponseGeneratorAgent:
    def __init__(self, model: str = OPENAI_CHAT_MODEL):
        self.model = model
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    # ------------------------------------------------------------------
    # Prompt builder — returns (prompt_str, included_sources)
    # ------------------------------------------------------------------
    def build_prompt(self,
                     query: str,
                     standalone_query: str,
                     sources: List[Dict[str, Any]],
                     detected_accent: str = "msa") -> Tuple[str, List[Dict[str, Any]]]:
        MAX_CONTEXT_CHARS = 9000
        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        seen_parent_ids = set()
        for src in sources:
            parent_id = src.get("parent_id") or src.get("chunk_id")
            if parent_id and parent_id in seen_parent_ids:
                continue

            score = src.get("similarity_score", 0)
            rank = len(included_sources) + 1
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = (src.get("parent_text") or src.get("text", "")).strip()
            if not raw_text:
                continue

            if parent_id:
                seen_parent_ids.add(parent_id)

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (درجة التوافق: {score})"

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
                "You are an Eng-Guild information assistant for the Jordan Engineers Association (JEA).\n"
                "Strict Response Guidelines (AGENTS.md):\n"
                "1. GIVE THE DIRECT ANSWER FIRST in sentence 1 at the very top. ZERO preamble, ZERO introductory filler, ZERO greeting.\n"
                "2. Mandatory Structure: Direct Answer -> Essential Details -> Actionable Next Steps.\n"
                "3. Short Sentences & Maximum Density: Write using short, concise sentences. Retain maximum factual detail while eliminating fluff.\n"
                "4. Language Matching: Respond strictly in professional English.\n"
                "5. NO Source Citations: Do NOT mention source tags or brackets like [Source 1] or [Source 2] inside your answer text.\n"
                "6. NO Chatbot Artifacts: NO generic sign-offs ('Let me know if you need anything else'), NO conversational filler.\n"
                "7. Accuracy: Do NOT invent facts. If missing, state: 'I couldn't find that information in the guild resources.'\n"
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
                f"5. عدم كتابة أرقام المصادر: يُمنع إدراج أرقام المصادر بين الأقواس مثل [المصدر 1] أو [المصدر N] داخل جمل الإجابة. اكتب الإجابة بسلاسة واحترافية وبدون رموز المصادر.\n"
                f"6. خلو من المظاهر الزائفة: يُحظر إضافة أي خاتمة روتينية أو جمل ترحيبية أو عرض مساعدة إضافية في النهاية.\n"
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

    def _to_messages(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Converts raw prompt string or pre-built messages into OpenAI message list."""
        if isinstance(prompt, list):
            return prompt
        if not isinstance(prompt, str) or not prompt.strip():
            return []

        parts = prompt.split("=" * 60)
        if len(parts) >= 2:
            system_part = parts[0].strip()
            user_part = ("=" * 60).join(parts[1:]).strip()
            return [
                {"role": "system", "content": system_part},
                {"role": "user", "content": user_part},
            ]
        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, bool]:
        messages = self._to_messages(prompt)
        if not messages:
            return "", False
        if not self._client:
            print("[OpenAI] Warning: OPENAI_API_KEY is not configured.")
            return "", False

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content:
                return clean_formatting(content), True
        except openai.AuthenticationError:
            print("[OpenAI] Authentication failed — check OPENAI_API_KEY.")
        except openai.RateLimitError:
            print("[OpenAI] Rate limit exceeded.")
        except openai.APIConnectionError as e:
            print(f"[OpenAI] Connection error: {e}")
        except Exception as e:
            print(f"[OpenAI] Unexpected error: {e}")

        return "", False

    # ------------------------------------------------------------------
    # Streaming generation (token-by-token, for SSE endpoint)
    # ------------------------------------------------------------------
    def generate_stream(self, prompt: Union[str, List[Dict[str, str]]]) -> Generator[str, None, None]:
        messages = self._to_messages(prompt)
        if not messages or not self._client:
            if not self._client:
                print("[OpenAI] Warning: OPENAI_API_KEY is not configured.")
            return

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                stream=True,
            )
            reasoning_filter = ReasoningStreamFilter()
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                if not token:
                    continue
                safe = reasoning_filter.feed(token)
                safe = safe.replace("*", "").replace("#", "")
                if safe:
                    yield safe
            tail = reasoning_filter.flush()
            tail = tail.replace("*", "").replace("#", "")
            if tail:
                yield tail
        except openai.AuthenticationError:
            print("[OpenAI] Authentication failed — check OPENAI_API_KEY.")
        except openai.RateLimitError:
            print("[OpenAI] Rate limit exceeded.")
        except openai.APIConnectionError as e:
            print(f"[OpenAI] Connection error: {e}")
        except Exception as e:
            print(f"[OpenAI] Unexpected error: {e}")
