from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from utils.llm_backend import ImagePromptLLM

DEFAULT_MODEL = 'gemini-3.5-flash-lite'
HUGGINFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


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


def get_hf_client():
    model = ImagePromptLLM(HUGGINFACE_MODEL)
    return model.load()
