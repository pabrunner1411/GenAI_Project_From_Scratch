import random

from .llm_client import BookAtmosphere

SYSTEM_INSTRUCTION = (
    'You are an expert art director for film adaptations. '
    'Extract the core visual and atmospheric tone of the literary text.'
)

JSON_INSTRUCTION = (
    'Respond with ONLY a single JSON object with these exact keys: '
    'overall_mood_keywords (list of 3 to 5 adjectives), '
    'visual_palette_and_lighting (string), '
    'narrative_tone_progression (string, one sentence).'
)


def get_stratified_book_samples(chunks, samples_per_section=2, seed=42):
    """Extracts representative chunks from the beginning, middle, and end of the book."""
    total_chunks = len(chunks)
    third = total_chunks // 3

    start_zone = chunks[:third]
    middle_zone = chunks[third:2 * third]
    end_zone = chunks[2 * third:]

    rng = random.Random(seed)
    start_samples = rng.sample(start_zone, min(samples_per_section, len(start_zone)))
    middle_samples = rng.sample(middle_zone, min(samples_per_section, len(middle_zone)))
    end_samples = rng.sample(end_zone, min(samples_per_section, len(end_zone)))

    formatted_sections = [
        '=== BEGINNING EXCERPTS ===\n' + '\n---\n'.join(start_samples),
        '=== MIDDLE EXCERPTS ===\n' + '\n---\n'.join(middle_samples),
        '=== ENDING EXCERPTS ===\n' + '\n---\n'.join(end_samples),
    ]

    return '\n\n' + ('=' * 40) + '\n\n'.join(formatted_sections)


def extract_book_atmosphere(client, chunks, samples_per_section=2, seed=42, temperature=0.2):
    book_samples = get_stratified_book_samples(chunks, samples_per_section=samples_per_section, seed=seed)

    prompt = f"""Analyze the provided excerpts taken from the start, middle, and end of a novel.
Synthesize the overarching visual atmosphere, mood, and aesthetic style of the story in a few concise words/sentences.

Excerpts:
{book_samples}

{JSON_INSTRUCTION}"""

    raw = client.generate_json(SYSTEM_INSTRUCTION, prompt, temperature=temperature)
    return BookAtmosphere.model_validate(raw).model_dump()
