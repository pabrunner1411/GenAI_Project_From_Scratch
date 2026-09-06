from pathlib import Path
from typing import Any

from .chunker import semantic_chunk_text
from .config import SUPPORTED_SUFFIXES
from .document_loader import extract_text
from .export_utils import export_results
from .text_cleaning import clean_text


class PipelineError(Exception):
    """Error that can be shown in a readable way in the UI."""


def process_document(
    uploaded_file: str | None,
    min_tokens: int,
    max_tokens: int,
    breakpoint_percentile: float,
    overlap_units: int,
    encoding_name: str,
    model_name: str,
    strip_gutenberg: bool,
) -> tuple[str, list[list[Any]], str, str, str, list[dict[str, Any]]]:
    """Runs loading, cleaning, chunking, and export."""
    if not uploaded_file:
        raise PipelineError("Please upload a file first.")

    source_path = Path(str(uploaded_file))

    if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise PipelineError(
            "Only TXT, MD, PDF, and DOCX files are allowed."
        )

    try:
        extracted_text = extract_text(source_path)
        cleaned_text = clean_text(
            extracted_text,
            strip_gutenberg=strip_gutenberg,
        )
    except Exception as error:
        raise PipelineError(
            f"The file could not be read: {error}"
        ) from error

    if len(cleaned_text) < 100:
        raise PipelineError(
            "Barely any text was found. "
            "A scanned PDF additionally requires OCR."
        )

    try:
        (
            chunks,
            total_tokens,
            semantic_threshold,
            atomic_unit_count,
        ) = semantic_chunk_text(
            text=cleaned_text,
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
            breakpoint_percentile=float(breakpoint_percentile),
            overlap_units=int(overlap_units),
            encoding_name=encoding_name,
            model_name=model_name,
        )
    except Exception as error:
        raise PipelineError(
            f"Semantic chunking failed: {error}"
        ) from error

    if not chunks:
        raise PipelineError("No chunks could be produced.")

    cleaned_path, jsonl_path = export_results(
        source_path=source_path,
        cleaned_text=cleaned_text,
        chunks=chunks,
    )

    preview_rows: list[list[Any]] = []

    for chunk in chunks[:20]:
        preview_text = chunk["text"].replace("\n", " ").strip()

        if len(preview_text) > 400:
            preview_text = preview_text[:400] + " …"

        preview_rows.append(
            [
                chunk["chunk_index"],
                chunk["token_count"],
                chunk["break_reason"],
                preview_text,
            ]
        )

    status = (
        "### Semantic chunking complete\n"
        f"- **File:** `{source_path.name}`\n"
        f"- **Characters:** {len(cleaned_text):,}\n"
        f"- **Tokens:** {total_tokens:,}\n"
        f"- **Atomic units:** {atomic_unit_count:,}\n"
        f"- **Chunks:** {len(chunks):,}\n"
        f"- **Semantic threshold:** {semantic_threshold:.3f}\n"
        f"- **Chunk size:** {int(min_tokens)}–{int(max_tokens)} tokens\n"
        f"- **Overlap:** {int(overlap_units)} unit(s)\n"
        f"- **Model:** `{model_name}`"
    )

    return status, preview_rows, cleaned_path, jsonl_path, cleaned_text, chunks
