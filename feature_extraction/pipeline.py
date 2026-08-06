import json
import tempfile
from pathlib import Path
from typing import Any

from .utils import (
    extract_book_atmosphere,
    extract_character_profiles,
    get_character_snippets,
    get_client,
    load_nlp,
    read_json,
    read_text,
    write_json,
)
from .utils.character_names import count_character_names
from .utils.llm_client import DEFAULT_MODEL


FEATURE_DIR = Path(__file__).resolve().parent
DATA_DIR = FEATURE_DIR / "data"
EXAMPLE_PROFILES_PATH = DATA_DIR / "extracted_character_profiles.json"
EXAMPLE_ATMOSPHERE_PATH = DATA_DIR / "book_atmosphere.json"


class FeatureExtractionError(Exception):
    """An error that can be displayed clearly in the Gradio interface."""


def load_example_data() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the Dracula example data bundled with Part 2."""
    profiles: dict[str, Any] = {}
    atmosphere: dict[str, Any] = {}

    if EXAMPLE_PROFILES_PATH.exists():
        profiles = read_json(EXAMPLE_PROFILES_PATH)

    if EXAMPLE_ATMOSPHERE_PATH.exists():
        atmosphere = read_json(EXAMPLE_ATMOSPHERE_PATH)

    return profiles, atmosphere


def _read_semantic_chunks(jsonl_path: Path) -> list[str]:
    """Read text fields from the JSONL chunks created by Part 1."""
    chunks: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FeatureExtractionError(
                    f"Invalid chunk file at line {line_number}: {error}"
                ) from error

            text = str(record.get("text", "")).strip()
            if text:
                chunks.append(text)

    return chunks


def _select_most_frequent_names(
    name_counts: dict[str, int],
    max_characters: int,
) -> dict[str, int]:
    """Limit API requests to the most frequently detected names."""
    if max_characters <= 0:
        raise FeatureExtractionError(
            "The maximum number of characters must be greater than 0."
        )

    return dict(list(name_counts.items())[:max_characters])


def extract_document_features(
    processing_state: dict[str, str] | None,
    name_threshold: int = 5,
    max_characters: int = 12,
    max_chunks_per_character: int = 3,
    nlp_model: str = "en_core_web_sm",
    gemini_model: str = DEFAULT_MODEL,
) -> tuple[str, dict[str, Any], dict[str, Any], str, str]:
    """
    Run Part 2 on the results produced by Part 1.

    The semantic chunks from Part 1 are reused as excerpts for character
    profiles and book-atmosphere extraction.
    """
    if not processing_state:
        raise FeatureExtractionError(
            "Please semantically chunk a document first."
        )

    cleaned_path = Path(processing_state.get("cleaned_path", ""))
    chunks_path = Path(processing_state.get("chunks_path", ""))
    document_name = processing_state.get("document_name", "Document")

    if not cleaned_path.is_file() or not chunks_path.is_file():
        raise FeatureExtractionError(
            "The intermediate results from Part 1 could not be found. "
            "Please semantically chunk the document again."
        )

    try:
        text = read_text(cleaned_path)
        chunks = _read_semantic_chunks(chunks_path)
    except FeatureExtractionError:
        raise
    except Exception as error:
        raise FeatureExtractionError(
            f"The chunking results could not be read: {error}"
        ) from error

    if not chunks:
        raise FeatureExtractionError(
            "The chunk file does not contain any usable text excerpts."
        )

    try:
        nlp = load_nlp(nlp_model)
    except OSError as error:
        raise FeatureExtractionError(
            f"The spaCy model '{nlp_model}' is not installed. "
            f"Run: python -m spacy download {nlp_model}"
        ) from error
    except Exception as error:
        raise FeatureExtractionError(
            f"The spaCy model could not be loaded: {error}"
        ) from error

    try:
        name_counts = count_character_names(
            text,
            nlp,
            threshold=int(name_threshold),
        )
    except Exception as error:
        raise FeatureExtractionError(
            f"Character names could not be detected: {error}"
        ) from error

    selected_names = _select_most_frequent_names(
        name_counts,
        int(max_characters),
    )

    if not selected_names:
        raise FeatureExtractionError(
            "spaCy did not detect any person names above the selected "
            "frequency threshold. Try lowering the name threshold."
        )

    snippets = get_character_snippets(chunks, selected_names.keys())

    try:
        client = get_client()
    except Exception as error:
        raise FeatureExtractionError(str(error)) from error

    profiles = extract_character_profiles(
        client=client,
        character_snippets=snippets,
        model=gemini_model,
        max_chunks=int(max_chunks_per_character),
    )

    if not profiles:
        raise FeatureExtractionError(
            "Gemini could not create a profile for any detected character."
        )

    try:
        atmosphere = extract_book_atmosphere(
            client=client,
            chunks=chunks,
            model=gemini_model,
        )
    except Exception as error:
        atmosphere = {
            "error": (
                "The character profiles were created, but the book "
                f"atmosphere could not be extracted: {error}"
            )
        }

    output_directory = Path(
        tempfile.mkdtemp(prefix="feature_extraction_")
    )
    profiles_path = output_directory / "character_profiles.json"
    atmosphere_path = output_directory / "book_atmosphere.json"

    write_json(profiles, profiles_path)
    write_json(atmosphere, atmosphere_path)

    status = (
        "### Feature extraction complete\n"
        f"- **Document:** `{document_name}`\n"
        f"- **Detected names above threshold:** {len(name_counts):,}\n"
        f"- **Names selected for Gemini:** {len(selected_names):,}\n"
        f"- **Character profiles created:** {len(profiles):,}\n"
        f"- **Gemini model:** `{gemini_model}`"
    )

    return (
        status,
        profiles,
        atmosphere,
        str(profiles_path),
        str(atmosphere_path),
    )
