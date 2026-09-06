"""Adapts character_extraction (Part 2) for use from the Part 1 Gradio app."""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .character_extraction import (
    count_character_names,
    extract_book_atmosphere,
    extract_character_profiles,
    get_character_snippets,
    get_client,
    load_nlp,
    resolve_title_references,
)

# .env may hold an optional HUGGINGFACEHUB_API_TOKEN (only needed for
# gated/private models or higher HF download rate limits; the default
# model is public). Preload it by explicit path since it lives outside
# app.py's CWD-relative .env search.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class FeatureExtractionError(Exception):
    """Error that can be shown in a readable way in the UI."""


_nlp = None
_client = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = load_nlp()
    return _nlp


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def unload_text_extraction_model():
    """Releases the cached Part 2 LLM client, if one was loaded."""
    global _client
    if _client is not None:
        _client.unload()
        _client = None


def extract_characters_and_atmosphere(
    cleaned_text: str | None,
    chunks: list[dict[str, Any]] | None,
    name_threshold: int,
    max_characters: int,
    max_chunks_per_character: int = 3,
) -> tuple[str, list[list[Any]], dict[str, Any], dict[str, Any]]:
    """Runs character and atmosphere extraction based on Part 1's output."""
    if not cleaned_text or not chunks:
        raise FeatureExtractionError(
            "Please chunk a text semantically in the section above first."
        )

    try:
        name_counts = count_character_names(
            cleaned_text, _get_nlp(), threshold=int(name_threshold)
        )
    except Exception as error:
        raise FeatureExtractionError(
            f"Name detection failed: {error}"
        ) from error

    if not name_counts:
        raise FeatureExtractionError(
            "No characters were found. "
            "Try a lower threshold."
        )

    chunk_texts = [chunk["text"] for chunk in chunks]
    name_counts, name_titles = resolve_title_references(
        chunk_texts, name_counts, return_title_map=True
    )

    top_names = list(name_counts.keys())[: int(max_characters)]

    try:
        snippets = get_character_snippets(chunk_texts, top_names, name_titles)
        client = _get_client()
        character_profiles = extract_character_profiles(
            client, snippets, name_counts, max_chunks=int(max_chunks_per_character)
        )
        book_atmosphere = extract_book_atmosphere(client, chunk_texts)
    except Exception as error:
        raise FeatureExtractionError(
            f"Extraction failed: {error}"
        ) from error

    for name, profile in character_profiles.items():
        profile["aliases"] = [
            f"the {title.title()}" for title in name_titles.get(name, [])
        ]

    name_rows = [
        [
            name,
            count,
            ", ".join(f"the {title.title()}" for title in name_titles.get(name, [])),
        ]
        for name, count in name_counts.items()
        if name in top_names
    ]

    status = (
        "### Character & atmosphere extraction complete\n"
        f"- **Detected characters:** {len(name_counts):,}\n"
        f"- **Profiled characters:** {len(character_profiles):,} "
        f"(limited to max. {int(max_characters)})\n"
        f"- **Minimum mentions:** {int(name_threshold)}"
    )

    return status, name_rows, character_profiles, book_atmosphere
