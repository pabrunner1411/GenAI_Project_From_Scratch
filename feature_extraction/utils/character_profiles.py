import json

from google.genai import types

from .llm_client import CharacterProfile, DEFAULT_MODEL

SYSTEM_INSTRUCTION = (
    "You are a lead character designer. "
    "Extract details based ONLY on the excerpts."
)


def extract_character_profiles(
    client,
    character_snippets,
    model=DEFAULT_MODEL,
    max_chunks=3,
    temperature=None,
):
    profiles = {}

    for name, chunks in character_snippets.items():
        if not chunks:
            continue

        formatted_context = "\n\n---\n\n".join(chunks[:max_chunks])
        prompt = f"Character Name: {name}\n\nExcerpts:\n{formatted_context}"

        config_kwargs = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": CharacterProfile,
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            profiles[name] = json.loads(response.text)
            print(f"Successfully processed: {name}")
        except Exception as error:
            print(f"Error processing {name}: {error}")

    return profiles
