"""Adapts image_generation (Part 4) for use from the Part 1 Gradio app."""

import re
from pathlib import Path

from .image_generation import ImageGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"


class ImageGenerationError(Exception):
    """Error that can be shown in a readable way in the UI."""


_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = ImageGenerator()
    return _generator


def unload_image_generation_model():
    """Releases the cached Part 4 diffusion model, if one was loaded."""
    global _generator
    if _generator is not None:
        _generator.unload()
        _generator = None


def _image_path(character_name: str) -> Path:
    """data/images/<character name>.png, overwritten on each new run."""
    slug = re.sub(r"[^a-z0-9]+", "_", character_name.lower()).strip("_")
    return IMAGES_DIR / f"{slug}.png"


def generate_character_images(image_prompts: dict[str, str]):
    """Renders one portrait per character, based on Part 3's output.

    Each portrait is also saved to data/images/<character name>.png,
    overwriting whatever was there from a previous run.
    """
    if not image_prompts:
        raise ImageGenerationError(
            "Please generate image prompts in the section above first."
        )

    generator = _get_generator()
    images = []

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for character_name, prompt in image_prompts.items():
        try:
            image = generator.generate_image(prompt)
            images.append((image, character_name))
            path = _image_path(character_name)
            image.save(path)
            print(f'Generated image: {character_name} -> {path}')
        except Exception as e:
            print(f"Error generating image for {character_name}: {e}")

    if not images:
        raise ImageGenerationError("No images could be generated.")

    status = (
        "### Image generation complete\n"
        f"- **Generated images:** {len(images):,} / {len(image_prompts):,}"
    )

    return status, images
