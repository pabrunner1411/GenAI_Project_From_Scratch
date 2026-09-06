import json
import os

from huggingface_hub import login
from pydantic import BaseModel

DEFAULT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'


class CharacterProfile(BaseModel):
    # Null when the excerpts don't describe the named character's own
    # appearance/pose/mood, rather than borrowing a more vividly described
    # bystander's. A character can be present and active in a scene
    # without ever being physically described themselves, especially a
    # first person narrator, who rarely describes their own appearance.
    physical_appearance: str | None = None
    personality_traits: list[str]
    signature_pose_or_expression: str | None = None
    vibe_and_mood: str
    creature_type: str | None = None
    attire: str | None = None
    age: str | None = None


class BookAtmosphere(BaseModel):
    overall_mood_keywords: list[str]  # 3 to 5 key adjectives
    visual_palette_and_lighting: str  # Color palette, shadows, atmospheric elements
    narrative_tone_progression: str  # One sentence describing how tone shifts from start to end


def _extract_json(text):
    """Best-effort extraction of a JSON object from free-form model output."""
    text = text.strip()

    if text.startswith('```'):
        text = text.strip('`')
        if text.lower().startswith('json'):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    return None


class LocalLLMClient:
    """Runs a small instruct model locally via transformers."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self._generator = None

        token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
        if token:
            login(token=token)

    def _get_generator(self):
        if self._generator is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._generator = pipeline(
                'text-generation', model=self.model_name, device=device
            )

        return self._generator

    def unload(self):
        """Releases the generation pipeline from memory, if one was loaded."""
        from .. import gpu_utils

        if self._generator is not None:
            self._generator = None
            gpu_utils.release_gpu_memory()
            print(f'Unloaded {self.model_name} (character extraction).')

    def generate_json(self, system_instruction, prompt, temperature=0.2, max_new_tokens=400):
        """Generates a response and parses it as JSON, retrying once on failure."""
        generator = self._get_generator()
        messages = [
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': prompt},
        ]

        raw_text = ''
        for _ in range(2):
            result = generator(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
            )
            raw_text = result[0]['generated_text']
            parsed = _extract_json(raw_text)
            if parsed is not None:
                return parsed

            messages[-1]['content'] += (
                '\n\nReminder: respond with ONLY a single valid JSON object, no other text.'
            )

        raise ValueError(f'Model did not return valid JSON after 2 attempts. Last output: {raw_text!r}')


def get_client(model_name=DEFAULT_MODEL):
    return LocalLLMClient(model_name)
