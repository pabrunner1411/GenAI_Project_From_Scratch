import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import spacy


def load_nlp(model_name="en_core_web_sm"):
    return spacy.load(model_name)


def clean_name(name):
    normalized = unicodedata.normalize("NFKD", name)
    cleaned = "".join(
        character
        for character in normalized
        if character.isalnum() or character.isspace()
    )
    return " ".join(cleaned.split()).strip()


def _split_for_spacy(text: str, max_length: int) -> list[str]:
    """Split very long documents without hard cuts in the middle of words."""
    if len(text) <= max_length:
        return [text]

    segments: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_length, len(text))

        if end < len(text):
            preferred_break = max(
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if preferred_break > start + max_length // 2:
                end = preferred_break + 1

        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end

    return segments


def _iter_docs(text: str, nlp) -> Iterable:
    safe_length = max(10_000, int(getattr(nlp, "max_length", 1_000_000)) - 1_000)
    segments = _split_for_spacy(text, safe_length)

    if len(segments) == 1:
        yield nlp(segments[0])
    else:
        yield from nlp.pipe(segments, batch_size=4)


def count_character_names(text, nlp, threshold=1):
    counts: Counter[str] = Counter()

    for doc in _iter_docs(text, nlp):
        names = [
            clean_name(entity.text).lower()
            for entity in doc.ents
            if entity.label_ == "PERSON"
        ]
        counts.update(name for name in names if name)

    filtered = {
        name: count
        for name, count in counts.items()
        if count >= threshold
    }

    return dict(
        sorted(
            filtered.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def get_character_snippets(chunks, character_names):
    names = list(character_names)
    snippets = {name: [] for name in names}
    patterns = {
        name: re.compile(
            rf"(?<!\w){re.escape(name)}(?!\w)",
            flags=re.IGNORECASE,
        )
        for name in names
    }

    for chunk in chunks:
        for name, pattern in patterns.items():
            if pattern.search(chunk):
                snippets[name].append(chunk)

    return snippets
