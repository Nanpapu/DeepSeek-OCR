from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ocr_pipeline.models import DeepSeekOCRConfig, DeepSeekOCRModel
from ocr_pipeline.postprocess import build_post_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on images and post-process the results."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more image paths or glob patterns.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional override prompt. Defaults to markdown conversion template.",
    )
    parser.add_argument(
        "--raw-output-dir",
        default="ocr_pipeline/outputs/raw",
        help="Directory to store raw OCR outputs.",
    )
    parser.add_argument(
        "--post-output-dir",
        default="ocr_pipeline/outputs/processed",
        help="Directory to store processed artifacts.",
    )
    parser.add_argument(
        "--post-processors",
        nargs="*",
        default=["markdown_to_excel"],
        help="List of post-processors to run (registered names).",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-OCR",
        help="HuggingFace model id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device override (e.g. "cuda" or "cpu"). Defaults to auto-detect.',
    )
    return parser.parse_args()


def expand_paths(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        expanded = list(Path().glob(pattern))
        if not expanded:
            candidate = Path(pattern)
            if candidate.exists():
                expanded = [candidate]
        paths.extend(expanded)
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError("No images matched the provided patterns.")
    return unique


def save_raw_text(raw_dir: Path, result) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{result.image_path.stem}.md"
    output_path.write_text(result.raw_text, encoding="utf-8")
    metadata_path = raw_dir / f"{result.image_path.stem}.json"
    metadata_path.write_text(
        json.dumps({"prompt": result.prompt}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    image_paths = expand_paths(args.images)

    model = DeepSeekOCRModel(
        DeepSeekOCRConfig(model_name=args.model_name, device=args.device)
    )
    raw_results = model.run_batch(
        image_paths,
        prompt=args.prompt,
        output_dir=args.raw_output_dir,
    )

    raw_dir = Path(args.raw_output_dir)
    post_root = Path(args.post_output_dir)

    for result in raw_results:
        raw_output_path = save_raw_text(raw_dir, result)
        context = {
            "source_name": result.image_path.stem,
            "raw_output_path": raw_output_path,
            "model_prompt": result.prompt,
        }

        for processor_name in args.post_processors:
            processor_dir = post_root / processor_name
            processor = build_post_processor(
                processor_name,
                output_dir=processor_dir,
            )
            processor.run(result.raw_text, context)


if __name__ == "__main__":
    main()
