from __future__ import annotations

from typing import Callable, Dict, Optional, Type

from .base import PostProcessor
from .markdown_to_excel import MarkdownToExcelProcessor

_REGISTRY: Dict[str, Type[PostProcessor]] = {
    MarkdownToExcelProcessor.name: MarkdownToExcelProcessor,
}


def register_post_processor(name: str, cls: Type[PostProcessor]) -> None:
    """Register a custom post-processor."""
    if name in _REGISTRY:
        raise ValueError(f"Post-processor '{name}' is already registered.")
    _REGISTRY[name] = cls


def build_post_processor(
    name: str,
    *,
    output_dir,
    **kwargs,
) -> PostProcessor:
    """Instantiate a post-processor by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Post-processor '{name}' is not registered.")
    cls = _REGISTRY[name]
    return cls(output_dir=output_dir, **kwargs)

