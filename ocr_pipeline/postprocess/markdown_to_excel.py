from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import PostProcessResult, PostProcessor


@dataclass
class MarkdownToExcelConfig:
    """Configuration for converting markdown/HTML tables to Excel."""

    excel_engine: Optional[str] = None
    table_index: int = 0
    filename_suffix: str = "_table"


class MarkdownToExcelProcessor(PostProcessor):
    """Convert a markdown or HTML table produced by DeepSeek-OCR into Excel."""

    name = "markdown_to_excel"

    def __init__(
        self,
        *,
        output_dir: Path | str,
        config: Optional[MarkdownToExcelConfig] = None,
    ) -> None:
        super().__init__(output_dir=output_dir)
        self.config = config or MarkdownToExcelConfig()

    def run(self, raw_text: str, context: Dict[str, Any]) -> PostProcessResult:  # noqa: D401
        import pandas as pd  # type: ignore

        table_frames = self._extract_tables(raw_text)
        if not table_frames:
            raise ValueError("No markdown/HTML tables detected in OCR output.")

        idx = min(self.config.table_index, len(table_frames) - 1)
        df = table_frames[idx]

        source_name = Path(context.get("source_name", "ocr_result")).stem
        output_path = self.output_dir / f"{source_name}{self.config.filename_suffix}.xlsx"
        df.to_excel(output_path, index=False, engine=self.config.excel_engine)

        metadata = {
            "rows": len(df),
            "columns": list(df.columns),
            "table_index": idx,
        }
        return PostProcessResult(
            input_path=Path(context.get("raw_output_path", "")) if context.get("raw_output_path") else Path(),
            output_path=output_path,
            metadata=metadata,
        )

    def _extract_tables(self, raw_text: str):
        import pandas as pd  # type: ignore

        tables: List[pd.DataFrame] = []

        html_tables = self._extract_html_tables(raw_text)
        if html_tables:
            tables.extend(html_tables)

        markdown_tables = self._extract_markdown_tables(raw_text)
        if markdown_tables:
            tables.extend(markdown_tables)

        return tables

    def _extract_html_tables(self, raw_text: str):
        import pandas as pd  # type: ignore

        if "<table" not in raw_text.lower():
            return []
        try:
            return pd.read_html(raw_text)
        except ValueError:
            return []

    def _extract_markdown_tables(self, raw_text: str):
        import pandas as pd  # type: ignore

        blocks = self._locate_markdown_blocks(raw_text.splitlines())
        frames = []
        for header, rows in blocks:
            frames.append(pd.DataFrame(rows, columns=header))
        return frames

    def _locate_markdown_blocks(
        self, lines: Sequence[str]
    ) -> List[tuple[List[str], List[List[str]]]]:
        blocks: List[List[str]] = []
        current: List[str] = []

        def flush():
            if current:
                blocks.append(current.copy())
                current.clear()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                current.append(stripped)
            else:
                flush()
        flush()

        parsed: List[tuple[List[str], List[List[str]]]] = []
        for block in blocks:
            if len(block) < 2:
                continue
            header = self._split_row(block[0])
            separator = self._split_row(block[1])
            if not all(cell.replace(":", "").replace("-", "") == "" for cell in separator):
                continue
            body_rows = [self._split_row(row) for row in block[2:] if row]
            if not body_rows:
                continue
            parsed.append((header, body_rows))
        return parsed

    @staticmethod
    def _split_row(row: str) -> List[str]:
        return [cell.strip() for cell in row.strip("|").split("|")]

