import json
import re
from typing import Any, Dict, Generator, List, Tuple

import requests

from src.config import NUM_CTX, NUM_PREDICT, OLLAMA_CHAT_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_URL

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
    cleaned = re.sub(r'\[\s*(المصدر|المصادر|Source|Sources)\s*[\d\.\, ]+\]', '', cleaned, flags=re.IGNORECASE)
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
                "7. Accuracy: Do NOT invent facts. If missing, state: 'I couldn't find that information in the guild resources.'\n\n"
                "Personal Attacks & De-escalation Rules:\n"
                "- If a message contains personal attacks or aggressive tone:\n"
                "  a) If purely abusive with no guild question: Reply calmly and professionally, clarifying your role as an information assistant.\n"
                "  b) If an insult is combined with a legitimate inquiry: Ignore the insult completely, extract the factual question, and answer directly.\n\n"
                "Few-Shot Demonstrations:\n"
                "---\n"
                "User Question: \"You are a useless stupid bot and your service is garbage\"\n"
                "Answer:\n"
                "I am an information assistant dedicated to the Jordan Engineers Association services, regulations, and procedures. How may I assist you with your guild inquiries today?\n"
                "---\n"
                "User Question: \"You thieves, why are you taking my money? How much is the annual renewal fee?\"\n"
                "Answer:\n"
                "The annual subscription fee for practicing engineers is 30 JOD per year.\n"
                "• Due Date: Payable at the beginning of each calendar year, with a grace period until the end of February.\n"
                "• Late Penalties: Late payment fines apply after the grace period per internal regulations.\n"
                "• Payment Methods: Online via eFAWATEERcom or at any JEA branch.\n"
                "---\n"
                "User Question: \"Your service is trash, I have been waiting a month for blueprint approval and nobody cares, where do I go?\"\n"
                "Answer:\n"
                "To track delayed engineering blueprint approvals and submit inquiries:\n"
                "• Online Tracking: Log in to the Engineering Offices Portal on the JEA website and review the Technical Review Committee report.\n"
                "• In-Person Inquiries: Visit the Technical Affairs Department at JEA Headquarters in Amman (3rd Floor).\n"
                "• Required Information: Provide the application transaction number and designer office name.\n"
                "---\n"
                "User Question: \"You are wasting our time, I have been waiting 2 months for center transfer, you failures what is the solution?\"\n"
                "Answer:\n"
                "To follow up on pending training or workplace center transfer requests:\n"
                "• Application Status: Log in to the electronic services portal under the Offices and Training section to check status.\n"
                "• Delay Expedited Review: Submit an expedited request via the online complaints window or visit the Training and Employment Division at JEA Headquarters.\n"
                "• Required Details: Provide your transaction number and JEA membership ID.\n"
                "---\n"
                "User Question: \"Your health insurance is garbage and covers nothing, how do I add my wife and kids?\"\n"
                "Answer:\n"
                "Adding family dependents (spouse and children) to the health insurance is processed during designated enrollment windows (January and July each year).\n"
                "• Required Documents: Copy of family book, personal photos for dependents, and proof of active annual membership.\n"
                "• Submission: Apply online via the Health Insurance portal or visit the nearest JEA branch.\n"
                "• Payment: Pay the applicable dependent insurance premium via eFAWATEERcom.\n"
                "---\n"
                "User Question: \"You ignorant people know nothing, where can I authenticate my engineering experience certificate for travel?\"\n"
                "Answer:\n"
                "Authentication of engineering experience certificates is conducted at the Technical and Professional Affairs Department at JEA Headquarters (Amman - Shmeisani) or governorate branches.\n"
                "• Requirements: The employing company or office must be officially registered and accredited by JEA, with experience matching subscription records.\n"
                "• Working Hours: Sunday through Thursday, 8:00 AM to 3:30 PM.\n"
                "---\n"
                "User Question: \"You crooks, where does the pension fund money go? What is the mandatory retirement age for engineers?\"\n"
                "Answer:\n"
                "The mandatory retirement age to qualify for full pension benefits from the JEA Pension Fund is:\n"
                "• 60 years for male engineers (with a minimum of 30 years / 360 months of paid contributions).\n"
                "• 55 years for female engineers (with a minimum of 25 years / 300 months of paid contributions).\n"
                "• Early Retirement: Available after age 55 for males and 50 for females with at least 25 years of paid contributions.\n"
                "• Application: Submit the pension application form through the online portal or visit the Pension Fund Administration.\n"
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
                f"7. الدقة والاعتماد على المصادر: اجب اعتماداً على المعلومات الواردة في المصادر المرفقة بأعلاه. إذا لم تتوفر المعلومة، اذكر بصراحة: 'لم أتمكن من العثور على هذه المعلومة في موارد النقابة المتاحة.'\n\n"
                f"قواعد التعامل مع الانفعال والإساءات الشخصية (Personal Attacks & De-escalation):\n"
                f"- في حال تضمنت الرسالة إساءات شخصية أو هجوماً أو ألفاظاً انفعالية:\n"
                f"  أ) إذا كانت الرسالة إساءة محضة بلا استفسار: أجب بهدوء ومهنية تامة دون انفعال أو دفاعية، ووضح دورك في خدمة المهندسين.\n"
                f"  ب) إذا تضمنت الرسالة إساءة مدمجة مع سؤال نقابي: تجاهل الإساءة تماماً، واستخرج السؤال الجوهري، وقدم الإجابة المباشرة والإجراءات فوراً.\n\n"
                f"أمثلة توضيحية (Few-Shot Demonstrations):\n"
                f"---\n"
                f"سؤال المستخدم: \"انت بوت غبي ومتخلف وما بتفهم اشي وحركتكم تافهة\"\n"
                f"الإجابة:\n"
                f"أنا مساعد معلومات مخصص لخدمات وأنظمة نقابة المهندسين الأردنيين. أنا هنا لمساعدتك في أي استفسار نقابي أو إجراء رسمي بكل مهنية واحترام. كيف يمكنني خدمتك؟\n"
                f"---\n"
                f"سؤال المستخدم: \"يا نصابين ليش بتسرقوا فلوسنا؟ قديش رسوم الاشتراك السنوي للمهندس؟\"\n"
                f"الإجابة:\n"
                f"رسم الاشتراك السنوي للمهندس الممارس هو 30 ديناراً أردنياً سنوياً.\n"
                f"• يستحق الرسم في بداية كل عام ميلادي، مع فترة سماح حتى نهاية شهر شباط (فبراير).\n"
                f"• بعد انتهاء فترة السماح تفرض غرامة تأخير وفق النظام الداخلي للنقابة.\n"
                f"• للدفع: متاح إلكترونياً عبر خدمة (إي فواتيركم) أو مباشرة في أي من فروع النقابة.\n"
                f"---\n"
                f"سؤال المستخدم: \"خدمتكم زبالة صرلي شهر بستنى مصادقة المخطط وما حدا معبرني بدي اعرف وين اروح\"\n"
                f"الإجابة:\n"
                f"لمتابعة حالة مصادقة المخططات الهندسية المتأخرة والاعتراض:\n"
                f"• الاستعلام الإلكتروني: ادخل إلى بوابة المكاتب والشركات الهندسية على موقع النقابة واطلع على تقرير لجنة التدقيق الفني.\n"
                f"• المراجعة المباشرة: راجع دائرة الشؤون الفنية بمجمع النقابات المهنية - عمان (الطابق الثالث).\n"
                f"• المتطلبات: إحضار رقم المعاملة واسم المكتب الهندسي المصمم.\n"
                f"---\n"
                f"سؤال المستخدم: \"بتتخوثوا علينا وصارلي شهرين بستنى معاملة نقل المركز وما حدا معبرني يا فاشلين شو الحل؟\"\n"
                f"الإجابة:\n"
                f"لمتابعة طلب نقل مركز التدريب أو العمل المعلق:\n"
                f"• الاستعلام والمتابعة: ادخل إلى حسابك في بوابة الخدمات الإلكترونية - قسم شؤون المكاتب والتدريب للتحقق من حالة الطلب.\n"
                f"• تقديم شكوى تأخير: يمكنك تقديم طلب تسريع أو شكوى عبر نافذة الشكاوى الإلكترونية أو مراجعة شعبة التدريب والتشغيل بالمقر الرئيسي.\n"
                f"• المتطلبات: تزويد الموظف المختص برقم المعاملة والرقم النقابي.\n"
                f"---\n"
                f"سؤال المستخدم: \"تأمينكم الصحي زفت وسرسرية وما بتغطوا اشي، كيف بدي اضيف زوجتي واولادي للتأمين؟\"\n"
                f"الإجابة:\n"
                f"إضافة أفراد العائلة (الزوج/الزوجة والأبناء) للتأمين الصحي يتم خلال فترات الاشتراك المحددة (شهر كانون الثاني وشهر تموز من كل عام).\n"
                f"• الوثائق المطلوبة: صورة دفتر العائلة، صور شخصية للمستفيدين، وإثبات سريان الاشتراك السنوي للمهندس.\n"
                f"• تقديم الطلب: عبر موقع النقابة الإلكتروني (قسم التأمين الصحي) أو بزيارة أقرب فرع للنقابة.\n"
                f"• الدفع: تسديد قسط التأمين المترتب عن التابعين عبر إي فواتيركم.\n"
                f"---\n"
                f"سؤال المستخدم: \"انتو ناس جهلة وما بتفهموا اشي، وين بقدر اصدق شهادة الخبرة الهندسية عشان السفر؟\"\n"
                f"الإجابة:\n"
                f"تصديق شهادات الخبرة الهندسية يتم لدى دائرة الشؤون الفنية والمهنية بمقر النقابة الرئيسي (عمان - الشميساني) أو في فروع المحافظات.\n"
                f"• الشروط المطلوبة: أن تكون جهة العمل (الشركة أو المكتب) مسجلة ومعتمدة لدى النقابة، ومطابقة فترة الخبرة لسجلات الاشتراك.\n"
                f"• الرسوم وأوقات الدوام: تدفع الرسوم المقررة للتصديق، وتستقبل الطلبات من الأحد إلى الخميس (8:00 صباحاً - 3:30 عصراً).\n"
                f"---\n"
                f"سؤال المستخدم: \"يا حرامية وين بتروح فلوس صندوق التقاعد؟ وشو سن التقاعد الإلزامي للمهندس؟\"\n"
                f"الإجابة:\n"
                f"سن التقاعد الإلزامي لاستحقاق الراتب التقاعدي الكامل من صندوق تقاعد نقابة المهندسين هو:\n"
                f"• 60 عاماً للمهندسين الذكور (مع إتمام مدة اشتراك لا تقل عن 30 عاماً / 360 شهراً).\n"
                f"• 55 عاماً للمهندسات الإناث (مع إتمام مدة اشتراك لا تقل عن 25 عاماً / 300 شهر).\n"
                f"• التقاعد المبكر: متاح بعد سن 55 للذكور و50 للإناث بشرط إتمام 25 عاماً من الاشتراك المسدد.\n"
                f"• لتقديم الطلب: تعبئة نموذج التقاعد عبر البوابة الإلكترونية أو زيارة إدارة صندوق التقاعد بالمقر الرئيسي.\n"
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
