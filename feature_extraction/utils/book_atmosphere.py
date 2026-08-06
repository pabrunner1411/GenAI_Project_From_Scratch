import json
import random

from google.genai import types

from .llm_client import BookAtmosphere, DEFAULT_MODEL

SYSTEM_INSTRUCTION = (
    "You are an expert art director for film adaptations. "
    "Extract the core visual and atmospheric tone of the literary text."
)


def get_stratified_book_samples(chunks, samples_per_section=2, seed=42):
    """Extracts representative chunks from the beginning, middle, and end."""
    if not chunks:
        raise ValueError("No chunks were provided.")

    total_chunks = len(chunks)
    third = max(1, total_chunks // 3)

    start_zone = chunks[:third]
    middle_zone = chunks[third:2 * third] or start_zone
    end_zone = chunks[2 * third:] or middle_zone

    rng = random.Random(seed)
    start_samples = rng.sample(
        start_zone,
        min(samples_per_section, len(start_zone)),
    )
    middle_samples = rng.sample(
        middle_zone,
        min(samples_per_section, len(middle_zone)),
    )
    end_samples = rng.sample(
        end_zone,
        min(samples_per_section, len(end_zone)),
    )

    formatted_sections = [
        "=== BEGINNING EXCERPTS ===\n" + "\n---\n".join(start_samples),
        "=== MIDDLE EXCERPTS ===\n" + "\n---\n".join(middle_samples),
        "=== ENDING EXCERPTS ===\n" + "\n---\n".join(end_samples),
    ]

    return "\n\n" + ("=" * 40) + "\n\n".join(formatted_sections)


def extract_book_atmosphere(
    client,
    chunks,
    samples_per_section=2,
    seed=42,
    model=DEFAULT_MODEL,
    temperature=None,
):
    book_samples = get_stratified_book_samples(
        chunks,
        samples_per_section=samples_per_section,
        seed=seed,
    )

    prompt = f"""Analyze the provided excerpts taken from the start, middle, and end of a novel.
Synthesize the overarching visual atmosphere, mood, and aesthetic style of the story in a few concise words/sentences.

Excerpts:
{book_samples}"""

    config_kwargs = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "response_mime_type": "application/json",
        "response_schema": BookAtmosphere,
    }
    if temperature is not None:
        config_kwargs["temperature"] = temperature

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    return json.loads(response.text)
