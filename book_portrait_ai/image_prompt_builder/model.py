# src/book_portrait_ai/model.py

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class CharacterProfile:
    """Represents a character profile with all extracted information."""
    book_id: str
    character_name: str
    description: str
    visual_traits: List[str]
    personality_traits: List[str]
    evidence: List[DocumentChunk]  
    prompt: str = ""  
    model_name: str = ""  