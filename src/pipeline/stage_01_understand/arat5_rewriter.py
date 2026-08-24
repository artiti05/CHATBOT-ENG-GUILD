import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import ARAT5_MODEL_NAME, ENABLE_ARAT5_REWRITER

TASK_PREFIX = "حول إلى الفصحى: "
MULTITURN_PREFIX = "دمج المحادثة: "


class AraT5DialectRewriter:
    """
    GPU-Accelerated Neural Dialect-to-MSA Rewriter & Multi-Turn Query Condensation Layer using fine-tuned AraT5.
    Translates Jordanian / Levantine dialectal Arabic into formal Modern Standard Arabic (MSA)
    and condenses conversational history into standalone search queries.
    """

    def __init__(self, model_name: str = ARAT5_MODEL_NAME, enable: bool = ENABLE_ARAT5_REWRITER):
        self.model_name = model_name
        self.enable = enable
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _lazy_load(self):
        if self.tokenizer is None and self.enable:
            try:
                print(f"[AraT5 GPU Rewriter] Loading model '{self.model_name}' on device '{self.device}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

                if self.device == "cuda":
                    # Use bfloat16 if supported (prevents T5 float16 NaN crashes)
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype
                    ).to(self.device)
                else:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

                self.model.eval()
                print(f"[AraT5 GPU Rewriter] Successfully loaded '{self.model_name}' on {self.device.upper()}.")
            except Exception as e:
                print(f"[AraT5 Warning] Could not load model '{self.model_name}': {e}. Falling back to Rule Normalization.")
                self.enable = False

    def rewrite_to_msa(self, dialect_text: str) -> str:
        """
        Translates a Jordanian dialect text query into formal Modern Standard Arabic (MSA).
        Returns clean MSA text string.
        """
        if not dialect_text or not dialect_text.strip() or not self.enable:
            return dialect_text

        self._lazy_load()
        if not self.model or not self.tokenizer:
            return dialect_text

        try:
            raw = dialect_text.strip()
            text_to_encode = TASK_PREFIX + raw if not raw.startswith(TASK_PREFIX) else raw
            inputs = self.tokenizer(text_to_encode, return_tensors="pt", max_length=128, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=1,
                    early_stopping=True
                )

            msa_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return msa_text if (msa_text and len(msa_text) >= 3) else dialect_text
        except Exception as err:
            print(f"[AraT5 Warning] Rewriting failed: {err}")
            return dialect_text

    def condense_multiturn(self, prior_context: str, current_query: str) -> str:
        """
        Uses AraT5 neural model to condense prior user context and follow-up query into a standalone query.
        If current_query is a new standalone topic, returns current_query.
        """
        if not current_query or not current_query.strip():
            return current_query or ""
        if not prior_context or not prior_context.strip() or not self.enable:
            return current_query.strip()

        self._lazy_load()
        if not self.model or not self.tokenizer:
            return current_query.strip()

        try:
            raw_input = f"{MULTITURN_PREFIX}{prior_context.strip()} | {current_query.strip()}"
            inputs = self.tokenizer(raw_input, return_tensors="pt", max_length=128, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=1,
                    early_stopping=True
                )

            condensed = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return condensed if (condensed and len(condensed) >= 3) else current_query.strip()
        except Exception as err:
            print(f"[AraT5 Warning] Multi-turn condensation failed: {err}")
            return current_query.strip()
