import sys
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import ARAT5_MODEL_NAME, ENABLE_ARAT5_REWRITER, USE_FP16


class AraT5DialectRewriter:
    """
    GPU-Accelerated Neural Dialect-to-MSA Rewriter Layer using AraT5 (UBC-NLP/arat5-base-dialect-msa).
    Translates Jordanian / Levantine dialectal Arabic into formal Modern Standard Arabic (MSA)
    for high-precision downstream hybrid search retrieval.
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
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                if self.device == "cuda" and USE_FP16:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
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
            inputs = self.tokenizer(dialect_text.strip(), return_tensors="pt", max_length=128, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )

            msa_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return msa_text if (msa_text and len(msa_text) >= 3) else dialect_text
        except Exception as err:
            print(f"[AraT5 Warning] Rewriting failed: {err}")
            return dialect_text
