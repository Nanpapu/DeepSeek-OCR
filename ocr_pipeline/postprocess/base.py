from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PostProcessResult:
    """Generic container for post-processing outputs."""

    input_path: Path
    output_path: Path
    metadata: Dict[str, Any]


class PostProcessor(ABC):
    """Abstract base class for post-processing blocks."""

    name: str

    def __init__(self, *, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self, raw_text: str, context: Dict[str, Any]) -> PostProcessResult:
        """Execute the transformation and return metadata about generated files."""

