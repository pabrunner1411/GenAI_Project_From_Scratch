from pathlib import Path

from docx import Document
from pypdf import PdfReader

from config import SUPPORTED_SUFFIXES


def read_plain_text(path: Path) -> str:
    """Read TXT and Markdown files using multiple encoding fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("The file's text encoding could not be detected.")


def read_pdf(path: Path) -> str:
    """Extract the existing text layer from a PDF file."""
    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text.strip()}")

    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    """Read all non-empty paragraphs from a DOCX file."""
    document = Document(str(path))

    return "\n\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


def extract_text(path: Path) -> str:
    """Select the appropriate loader based on the file extension."""
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: {allowed}"
        )

    if suffix in {".txt", ".md"}:
        return read_plain_text(path)

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    raise ValueError("The file could not be processed.")
