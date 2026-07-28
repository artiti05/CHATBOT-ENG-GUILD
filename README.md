# Arabic PDF Parser (Qwen2.5-VL + Ollama)

Parses Arabic/English PDFs — including tables with merged/nested RTL headers —
by rendering each page to an image, detecting and isolating tables with
OpenCV, and using Qwen2.5-VL (via a local Ollama server) to transcribe prose
and parse tables separately.

## Why this approach

Text-layer extraction from Arabic PDFs is often unreliable (RTL reordering,
ligatures, embedded font encoding issues), especially in tables. This
pipeline treats each page as an image and uses a vision-language model
instead, with a specific fix for a common VLM failure mode: tables with a
merged header spanning several RTL sub-columns tend to get flattened or
garbled if the whole page is parsed in a single pass. This repo detects
table regions, crops and upscales them, and parses them in a dedicated
request so the model isn't juggling prose and complex table structure at
the same time.

## Pipeline

1. **Render** — each PDF page → PNG at 300 DPI (PyMuPDF)
2. **Preprocess** — grayscale + CLAHE (adaptive contrast; preserves faint
   borders/shading, unlike hard binarization)
3. **Detect tables** — OpenCV line-morphology finds bordered table regions
4. **Mask + prose pass** — tables are painted over with a placeholder token
   (`[[TABLE_1]]`, etc.) and the page is sent to Qwen for prose transcription
5. **Table pass** — each table region is cropped from the *unmasked* image,
   upscaled, and sent to Qwen separately with an RTL-merged-header-aware
   prompt, requesting clean HTML output
6. **Splice** — table HTML is substituted back into the prose text at each
   placeholder token

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- The vision model pulled:
  ```bash
  ollama pull qwen2.5vl:7b
  ```

## Setup

```bash
git clone https://github.com/<your-username>/arabic-pdf-parser.git
cd arabic-pdf-parser
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# start Ollama if it isn't already running
ollama serve

# run the parser
python arabic_pdf_parser.py path/to/input.pdf path/to/output_dir
```

Output structure:
```
output_dir/
├── raw_pages/                    # rendered page images
├── processed_pages/              # grayscale + CLAHE images
├── debug_detected_tables/        # pages with red boxes over detected tables
├── page_text/                    # per-page parsed .md files
└── <pdf_name>_parsed.md          # combined final output
```

## Configuration

Key constants at the top of `arabic_pdf_parser.py`:

| Variable | Purpose |
|---|---|
| `MODEL_NAME` | Ollama model tag (must match `ollama list` exactly) |
| `RENDER_DPI` | Page render resolution (250–300 recommended) |
| `NUM_CTX` | Ollama context window — raise if you see `exceed_context_size_error` |
| `MIN_TABLE_AREA_FRACTION` | Minimum size (as % of page) to count as a detected table |
| `TABLE_UPSCALE_FACTOR` | How much to enlarge a cropped table before parsing |

## Troubleshooting

- **`exceed_context_size_error`** → raise `NUM_CTX` (image tokens + prompt can
  exceed Ollama's default 4096 context window)
- **Missing or false-positive table detections** → check
  `debug_detected_tables/`, then tune `MIN_TABLE_AREA_FRACTION`
- **Placeholder token not found in prose output** → the model paraphrased the
  token instead of echoing it verbatim; the script appends the table HTML at
  the end instead of dropping it, but check the source page's layout

## License

MIT (or your preference)
