import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

DEFAULT_MODEL = "gemini-3.5-flash-lite"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class CharacterProfile(BaseModel):
    physical_appearance: str
    personality_traits: list[str]
    signature_pose_or_expression: str
    vibe_and_mood: str


class BookAtmosphere(BaseModel):
    overall_mood_keywords: list[str]
    visual_palette_and_lighting: str
    narrative_tone_progression: str


def get_client():
    """Load the API key from feature_extraction/.env and create the client."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy feature_extraction/.env.example "
            "to feature_extraction/.env and enter your API key there."
        )

    return genai.Client(api_key=api_key)
