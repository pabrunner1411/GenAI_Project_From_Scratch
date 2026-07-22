from pathlib import Path

from docx import Document
from pypdf import PdfReader

from config import SUPPORTED_SUFFIXES


def read_plain_text(path: Path) -> str:
    """Liest TXT- und Markdown-Dateien mit mehreren Encoding-Fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Die Textkodierung der Datei konnte nicht erkannt werden.")


def read_pdf(path: Path) -> str:
    """Extrahiert die vorhandene Textebene aus einer PDF-Datei."""
    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(f"[Seite {page_number}]\n{page_text.strip()}")

    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    """Liest alle nicht leeren Absätze aus einer DOCX-Datei."""
    document = Document(str(path))

    return "\n\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


def extract_text(path: Path) -> str:
    """Wählt anhand der Dateiendung den passenden Loader."""
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Nicht unterstütztes Dateiformat: {suffix}. Erlaubt: {allowed}"
        )

    if suffix in {".txt", ".md"}:
        return read_plain_text(path)

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    raise ValueError("Die Datei konnte nicht verarbeitet werden.")
