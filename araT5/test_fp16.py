"""
Run this BEFORE training anything.

It loads AraT5 three ways (fp32, bfloat16, float16) and prints what each one
outputs for the same Arabic query.

What you are looking for:
  - If the float16 column is empty, or repeated symbols, or nonsense
    while fp32 produces something readable -> fp16 is your bug.
  - If ALL THREE produce nonsense -> the model is simply untrained for this
    task (expected for AraT5-base), and fine-tuning is the fix.

Either way you learn something concrete in 10 minutes.

Usage:  python test_fp16.py
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "UBC-NLP/AraT5-base"        # the model your config currently uses

QUERIES = [
    "شو الأوراق المطلوبة عشان أسجل بالنقابة؟",
    "قديش رسم الاشتراك السنوي؟",
    "كيف بقدر اشترك بصندوق التقاعد؟",
]


def run(dtype, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        if dtype is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        for q in QUERIES:
            inputs = tok(q, return_tensors="pt", max_length=128,
                         truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=64, num_beams=1)
            text = tok.decode(out[0], skip_special_tokens=True)

            # a NaN check: if the encoder overflowed, this will show it
            with torch.no_grad():
                enc = model.get_encoder()(**inputs).last_hidden_state
            has_nan = torch.isnan(enc).any().item()

            print(f"  in : {q}")
            print(f"  out: {text!r}")
            print(f"  NaN in encoder: {has_nan}")
            print()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    run(None, "float32  (baseline)")
    run(torch.bfloat16, "bfloat16 (recommended)")
    run(torch.float16, "float16  (what your code uses now)")
