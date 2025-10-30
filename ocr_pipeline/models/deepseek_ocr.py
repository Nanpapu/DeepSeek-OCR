from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DeepSeekOCRConfig:
    """Configuration for loading and running DeepSeek-OCR."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    device: Optional[str] = None  # "cuda", "cpu" or None to auto-detect
    attn_implementation: str = "flash_attention_2"
    use_safetensors: bool = True
    prompt_template: str = "<image>\\n<|grounding|>Convert the document to markdown."
    default_base_size: int = 1024
    default_image_size: int = 640
    default_crop_mode: bool = True
    default_save_results: bool = True
    default_test_compress: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    """Container for the raw output produced by the model."""

    image_path: Path
    prompt: str
    raw_text: str
    extra: Dict[str, Any] = field(default_factory=dict)


class DeepSeekOCRModel:
    """Thin wrapper encapsulating tokenizer/model loading and inference."""

    def __init__(self, config: Optional[DeepSeekOCRConfig] = None) -> None:
        self.config = config or DeepSeekOCRConfig()
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load tokenizer and model weights into memory."""
        from transformers import AutoModel, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_safetensors=self.config.use_safetensors,
            _attn_implementation=self.config.attn_implementation,
            **self.config.model_kwargs,
        )

        self._model = self._model.eval()

        target_dtype = torch.bfloat16 if self.config.device == "cuda" else None
        if target_dtype is not None:
            self._model = self._model.to(self.config.device).to(target_dtype)
        else:
            self._model = self._model.to(self.config.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def run_batch(
        self,
        image_paths: Iterable[Path | str],
        prompt: Optional[str] = None,
        *,
        output_dir: Optional[Path | str] = None,
        base_size: Optional[int] = None,
        image_size: Optional[int] = None,
        crop_mode: Optional[bool] = None,
        save_results: Optional[bool] = None,
        test_compress: Optional[bool] = None,
        infer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[OCRResult]:
        """Run inference on a batch of images and return the raw outputs."""
        self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        resolved_prompt = prompt or self.config.prompt_template
        out_dir = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        results: List[OCRResult] = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            kwargs = {
                "tokenizer": self._tokenizer,
                "prompt": resolved_prompt,
                "image_file": str(path),
                "output_path": str(out_dir) if out_dir else None,
                "base_size": base_size or self.config.default_base_size,
                "image_size": image_size or self.config.default_image_size,
                "crop_mode": (
                    self.config.default_crop_mode if crop_mode is None else crop_mode
                ),
                "save_results": (
                    self.config.default_save_results
                    if save_results is None
                    else save_results
                ),
                "test_compress": (
                    self.config.default_test_compress
                    if test_compress is None
                    else test_compress
                ),
            }

            if infer_kwargs:
                kwargs.update(infer_kwargs)

            raw_output = self._infer_single(**kwargs)
            results.append(
                OCRResult(
                    image_path=path,
                    prompt=resolved_prompt,
                    raw_text=raw_output.get("text", raw_output.get("raw", "")),
                    extra=raw_output,
                )
            )
        return results

    def _infer_single(self, **kwargs: Any) -> Dict[str, Any]:
        """Delegate to the model-specific inference API."""
        assert self._model is not None

        if hasattr(self._model, "infer"):
            return self._model.infer(**kwargs)

        # Fallback: use generate interface
        tokenizer = kwargs["tokenizer"]
        prompt = kwargs["prompt"]
        image_file = kwargs["image_file"]
        from PIL import Image  # type: ignore
        from transformers import AutoProcessor  # type: ignore

        image = Image.open(image_file).convert("RGB")
        processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        generated_ids = self._model.generate(**inputs, max_new_tokens=2048)
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return {"text": output_text}
