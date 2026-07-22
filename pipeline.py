from pathlib import Path
from typing import Any

from config import SUPPORTED_SUFFIXES
from document_loader import extract_text
from export_utils import export_results
from semantic_chunking import semantic_chunk_text
from text_cleaning import clean_text


class PipelineError(Exception):
    """Fehler, der verständlich in der Oberfläche angezeigt werden kann."""


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
    """Führt Laden, Bereinigung, Chunking und Export aus."""
    if not uploaded_file:
        raise PipelineError("Bitte zuerst eine Datei hochladen.")

    source_path = Path(str(uploaded_file))

    if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise PipelineError(
            "Erlaubt sind nur TXT-, MD-, PDF- und DOCX-Dateien."
        )

    try:
        extracted_text = extract_text(source_path)
        cleaned_text = clean_text(
            extracted_text,
            strip_gutenberg=strip_gutenberg,
        )
    except Exception as error:
        raise PipelineError(
            f"Die Datei konnte nicht gelesen werden: {error}"
        ) from error

    if len(cleaned_text) < 100:
        raise PipelineError(
            "Es wurde kaum Text gefunden. "
            "Bei einer eingescannten PDF wird zusätzlich OCR benötigt."
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
            f"Semantic Chunking fehlgeschlagen: {error}"
        ) from error

    if not chunks:
        raise PipelineError("Es konnten keine Chunks erzeugt werden.")

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
        "### Semantic Chunking abgeschlossen\n"
        f"- **Datei:** `{source_path.name}`\n"
        f"- **Zeichen:** {len(cleaned_text):,}\n"
        f"- **Tokens:** {total_tokens:,}\n"
        f"- **Atomare Einheiten:** {atomic_unit_count:,}\n"
        f"- **Chunks:** {len(chunks):,}\n"
        f"- **Semantischer Grenzwert:** {semantic_threshold:.3f}\n"
        f"- **Chunkgröße:** {int(min_tokens)}–{int(max_tokens)} Tokens\n"
        f"- **Overlap:** {int(overlap_units)} Einheit(en)\n"
        f"- **Modell:** `{model_name}`"
    )

    return status, preview_rows, cleaned_path, jsonl_path
