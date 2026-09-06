"""Adapts image_prompt_builder (Part 3) for use from the Part 1 Gradio app."""

import json
import tempfile
from pathlib import Path

from .image_prompt_builder import ImagePromptBuilder


class ImagePromptGenerationError(Exception):
    """Error that can be shown in a readable way in the UI."""


_builder = None


def _get_builder():
    global _builder
    if _builder is None:
        _builder = ImagePromptBuilder()
    return _builder


def unload_image_prompt_model():
    """Releases the cached Part 3 image prompt LLM, if one was loaded."""
    global _builder
    if _builder is not None:
        _builder.unload()
        _builder = None


def _build_character_data(name, profile, book_atmosphere):
    return {
        "character_name": name.title(),
        "description": profile["physical_appearance"],
        "visual_traits": [],
        "personality_traits": profile["personality_traits"],
        "signature_pose_or_expression": profile["signature_pose_or_expression"],
        "vibe_and_mood": profile["vibe_and_mood"],
        # None when the book excerpts didn't describe it. The builder
        # leaves these out of the prompt entirely rather than saying
        # "unspecified", to keep the small model from treating that as
        # a real detail to write about.
        "creature_type": profile.get("creature_type"),
        "attire": profile.get("attire"),
        "age": profile.get("age"),
        "book_aesthetic": {
            "mood": ", ".join(book_atmosphere["overall_mood_keywords"]),
            "setting": book_atmosphere["visual_palette_and_lighting"],
        },
    }


def generate_image_prompts(character_profiles, book_atmosphere):
    """Generates one image prompt per profiled character, based on Part 2's output."""
    if not character_profiles or not book_atmosphere:
        raise ImagePromptGenerationError(
            "Please extract characters & atmosphere in the section above first."
        )

    builder = _get_builder()
    prompts = {}

    for name, profile in character_profiles.items():
        character_data = _build_character_data(name, profile, book_atmosphere)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as temp_file:
            json.dump(character_data, temp_file)
            temp_path = temp_file.name

        try:
            prompts[character_data["character_name"]] = builder.generate_image_prompt(temp_path)
        except Exception as e:
            print(f"Error generating image prompt for {name}: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    if not prompts:
        raise ImagePromptGenerationError("No image prompts could be generated.")

    status = (
        "### Image prompt generation complete\n"
        f"- **Generated prompts:** {len(prompts):,} / {len(character_profiles):,}"
    )

    return status, prompts
