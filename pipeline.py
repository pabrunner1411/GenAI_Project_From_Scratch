from pathlib import Path
from typing import Any

from config import SUPPORTED_SUFFIXES
from document_loader import extract_text
from export_utils import export_results
from semantic_chunking import semantic_chunk_text
from text_cleaning import clean_text


class PipelineError(Exception):
    """An error that can be displayed clearly in the user interface."""


def _process_document(
    uploaded_file: str | None,
    min_tokens: int,
    max_tokens: int,
    breakpoint_percentile: float,
    overlap_units: int,
    encoding_name: str,
    model_name: str,
    strip_gutenberg: bool,
) -> tuple[str, list[list[Any]], str, str, dict[str, str]]:
    """Shared internal implementation for Part 1 and the integration."""
    if not uploaded_file:
        raise PipelineError("Please upload a file first.")

    source_path = Path(str(uploaded_file))

    if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise PipelineError(
            "Only TXT, MD, PDF, and DOCX files are supported."
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
            "Very little text was found. "
            "A scanned PDF requires an additional OCR step."
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
        raise PipelineError("No chunks could be created.")

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
        f"- **Chunk size:** {int(min_tokens)}–{int(max_tokens)} tokens"
    )

    processing_state = {
        "document_name": source_path.name,
        "source_path": str(source_path),
        "cleaned_path": cleaned_path,
        "chunks_path": jsonl_path,
    }

    return (
        status,
        preview_rows,
        cleaned_path,
        jsonl_path,
        processing_state,
    )


def process_document(
    uploaded_file: str | None,
    min_tokens: int,
    max_tokens: int,
    breakpoint_percentile: float,
    overlap_units: int,
    encoding_name: str,
    model_name: str,
    strip_gutenberg: bool,
) -> tuple[str, list[list[Any]], str, str]:
    """Preserve the original Part 1 interface with four outputs."""
    result = _process_document(
        uploaded_file=uploaded_file,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        breakpoint_percentile=breakpoint_percentile,
        overlap_units=overlap_units,
        encoding_name=encoding_name,
        model_name=model_name,
        strip_gutenberg=strip_gutenberg,
    )
    return result[:4]


def process_document_with_state(
    uploaded_file: str | None,
    min_tokens: int,
    max_tokens: int,
    breakpoint_percentile: float,
    overlap_units: int,
    encoding_name: str,
    model_name: str,
    strip_gutenberg: bool,
) -> tuple[str, list[list[Any]], str, str, dict[str, str]]:
    """Extend Part 1 with file paths required by Part 2."""
    return _process_document(
        uploaded_file=uploaded_file,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        breakpoint_percentile=breakpoint_percentile,
        overlap_units=overlap_units,
        encoding_name=encoding_name,
        model_name=model_name,
        strip_gutenberg=strip_gutenberg,
    )
