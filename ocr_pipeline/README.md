# OCR Pipeline

This folder hosts a self-contained DeepSeek-OCR pipeline with three clear layers:

- `models/`: wrappers for loading & running DeepSeek-OCR.
- `postprocess/`: utilities that transform raw model output (e.g., markdown table ➝ Excel).
- `scripts/`: runnable entry points that orchestrate inference and post-processing.
- `images/`: drop your input images here (or use any other directory via CLI args).
- `outputs/`: split into `raw/` (model text dumps) and `processed/` (post-processing artifacts).

## Quick Start
1. Install dependencies (run inside your DeepSeek-OCR conda env). The repo-wide `requirements.txt` now bundles both the base project and pipeline extras such as pandas/openpyxl:
   ```bash
   pip install -r requirements.txt
   ```
2. Place test images inside `ocr_pipeline/images/`.
3. Run the pipeline:
   ```bash
   python ocr_pipeline/scripts/run_pipeline.py --images "ocr_pipeline/images/*.png"
   ```

The script will:
- Load `deepseek-ai/DeepSeek-OCR` (customise with `--model-name` if needed).
- Generate markdown output per image and save to `outputs/raw/<image>.md`.
- Apply the `markdown_to_excel` post-processor, producing `outputs/processed/markdown_to_excel/<image>_table.xlsx`.

## Customising Prompts & Post-Processing
- Override the prompt:
  ```bash
  python ocr_pipeline/scripts/run_pipeline.py --images "..." --prompt "<image>\n<|grounding|>OCR this image."
  ```
- Chain multiple processors (register new ones in `postprocess/registry.py`):
  ```bash
  python ocr_pipeline/scripts/run_pipeline.py --images "..." --post-processors markdown_to_excel another_processor
  ```
- Change output locations with `--raw-output-dir` or `--post-output-dir`.

## Extending
- Add new processors by subclassing `PostProcessor` in `postprocess/base.py`, then `register_post_processor` in `postprocess/registry.py`.
- Implement alternative prompts or batching logic by extending `DeepSeekOCRModel` in `models/deepseek_ocr.py`.

This structure keeps model inference code, raw outputs, and downstream transformations neatly separated so you can plug in new workflows (e.g., markdown ➝ database, HTML ➝ PDF) without touching the core OCR wrapper.
