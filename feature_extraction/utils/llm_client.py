from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

DEFAULT_MODEL = 'gemini-3.5-flash-lite'


class CharacterProfile(BaseModel):
    physical_appearance: str
    personality_traits: list[str]
    signature_pose_or_expression: str
    vibe_and_mood: str


class BookAtmosphere(BaseModel):
    overall_mood_keywords: list[str]  # 3 to 5 key adjectives
    visual_palette_and_lighting: str  # Color palette, shadows, atmospheric elements
    narrative_tone_progression: str  # One sentence describing how tone shifts from start to end


def get_client():
    load_dotenv()
    return genai.Client()
