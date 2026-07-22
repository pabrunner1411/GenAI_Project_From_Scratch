import json
import re
import tempfile
from pathlib import Path
from typing import Any


def safe_filename(value: str) -> str:
    """Erzeugt einen einfachen, sicheren Dateinamen."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "document"


def export_results(
    source_path: Path,
    cleaned_text: str,
    chunks: list[dict[str, Any]],
) -> tuple[str, str]:
    """Exportiert den bereinigten Text und die Chunks als JSONL."""
    output_directory = Path(
        tempfile.mkdtemp(prefix="semantic_chunking_")
    )

    document_name = safe_filename(source_path.stem)

    cleaned_path = output_directory / f"{document_name}_cleaned.txt"
    jsonl_path = output_directory / f"{document_name}_chunks.jsonl"

    cleaned_path.write_text(cleaned_text, encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "id": f"{document_name}-{chunk['chunk_index']:05d}",
                "document": source_path.name,
                **chunk,
            }
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return str(cleaned_path), str(jsonl_path)
