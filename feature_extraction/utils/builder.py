import json
from typing import Dict, Any, Optional


class ImagePromptBuilder:
    """
    Builds image generation prompts from character JSON data.
    
    """
    
    # Default artistic style prompt for consistency across all characters
    DEFAULT_STYLE_PROMPT = (
        "digital illustration, storybook art style, warm color palette"
    )
    
    # Paths to data files
    CHARACTER_PROFILES_PATH = "data/extracted_character_profiles.json"
    BOOK_ATMOSPHERE_PATH = "data/book_atmosphere.json"
    
    
    def __init__(self, 
                 llm = None,
                 style_prompt: Optional[str] = None):
        """
        Initialize the prompt builder.
        
        Args:
            style_prompt: Optional custom style prompt for consistency.
                         If None, uses the default.
        """
        self.style_prompt = style_prompt or self.DEFAULT_STYLE_PROMPT
        self.llm = llm
        self.character_profiles = None
        self.book_atmosphere = None
    

    def _load_character_profiles(self) -> Dict[str, Any]:
        """
        Load character profiles from JSON file.
        
        Returns:
            Dictionary of character profiles
        """
        if self.character_profiles is None:
            try:
                with open(self.CHARACTER_PROFILES_PATH, "r", encoding='utf-8') as file:
                    self.character_profiles = json.load(file)
            except FileNotFoundError:
                raise ValueError(f"Character profiles file not found: {self.CHARACTER_PROFILES_PATH}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in file: {self.CHARACTER_PROFILES_PATH}")
        
        return self.character_profiles
    
    
    def _load_book_atmosphere(self) -> Dict[str, Any]:
        """
        Load book atmosphere from JSON file.
        
        Returns:
            Dictionary of book atmosphere data
        """
        if self.book_atmosphere is None:
            try:
                with open(self.BOOK_ATMOSPHERE_PATH, "r", encoding='utf-8') as file:
                    self.book_atmosphere = json.load(file)
            except FileNotFoundError:
                raise ValueError(f"Book atmosphere file not found: {self.BOOK_ATMOSPHERE_PATH}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in file: {self.BOOK_ATMOSPHERE_PATH}")
        
        return self.book_atmosphere
    
    
    def _get_character_data(self, character_name: str) -> Dict[str, Any]:
        """
        Get character data for a specific character.
        
        Args:
            character_name: Name of the character to retrieve
            
        Returns:
            Character data dictionary
        """
        profiles = self._load_character_profiles()
        
        # Normalize character name for lookup 
        character_name_lower = character_name.lower()
        
        # Get match
        if character_name_lower in profiles:
            return profiles[character_name_lower]
        else:
            raise ValueError(f"Character '{character_name}' not found in profiles. Available characters: {', '.join(profiles.keys())}")
    
    
    def _build_character_prompt(self, character_data: Dict[str, Any]) -> str:
        """
        Build a descriptive image prompt from character JSON data.
        Creates the character-specific prompt.
        
        Expected JSON structure:
        {
            "physical_appearance": "Text description of appearance",
            "personality_traits": ["trait1", "trait2"],
            "signature_pose_or_expression": "Description of signature pose",
            "vibe_and_mood": "Overall vibe and mood of the character"
        }
        """
        character_prompt = (
            f"Physical Appearance: {character_data.get('physical_appearance', '')}\n"
            f"Personality Traits: {', '.join(character_data.get('personality_traits', []))}\n"
            f"Signature Pose or Expression: {character_data.get('signature_pose_or_expression', '')}\n"
            f"Vibe and Mood: {character_data.get('vibe_and_mood', '')}\n"
        )
        
        return character_prompt
    
    
    def _build_book_atmosphere_prompt(self) -> str:
        """
        Build a descriptive prompt from book atmosphere JSON data.

        Expected JSON structure:
                {
                    "overall_mood_keywords": ["key1", "key2"],
                    "visual_palette_and_lighting": "str",
                    "narrative_tone_progression": "str"
                }
        """
        atmosphere = self._load_book_atmosphere()
        
        atmosphere_prompt = (
            f"Overall Mood Keywords: {', '.join(atmosphere.get('overall_mood_keywords', []))}\n"
            f"Visual Palette and Lighting: {atmosphere.get('visual_palette_and_lighting', '')}\n"
            f"Narrative Tone Progression: {atmosphere.get('narrative_tone_progression', '')}\n"
        )
        
        return atmosphere_prompt
    
    
    def _build_llm_prompt(self, 
                          character_data: Dict[str, Any], 
                          character_name: str) -> str:
        """
        Build the prompt to send to the LLM for enhancement.
        Combines character prompt with book atmosphere and style consistency.
        
        Args:
            character_data: Character JSON dictionary
            character_name: Name of the character
            
        Returns:
            LLM prompt string
        """
        # Build the character-specific prompt
        character_prompt = self._build_character_prompt(character_data)
        
        # Build the book atmosphere prompt
        atmosphere_prompt = self._build_book_atmosphere_prompt()
        
        # Combine all elements
        combined_prompt = (
            f"Character Name: {character_name}\n"
            f"{character_prompt}\n"
            f"Book Atmosphere:\n{atmosphere_prompt}\n"
            f"Image Artistic Style: {self.style_prompt}"
        )
        
        llm_prompt = (
            "You are an expert prompt engineer for AI image generation. Your task is to create a single, "
            "coherent, descriptive prompt for a text-to-image diffusion model.\n\n"
            "Based on the following character information, book atmosphere, and artistic style, "
            "create a detailed image generation prompt.\n\n"
            
            "CHARACTER AND BOOK INFORMATION:\n"
            f"{combined_prompt}\n\n"
            
            "INSTRUCTIONS:\n"
            "- Create ONLY the image prompt text, nothing else\n"
            "- Do not include code, explanations, or extra text\n"
            "- Focus on visual details, lighting, composition, and mood\n"
            "- The prompt should be about 2-4 sentences\n"
            "- Start with the character name and key visual features\n"
            "- Incorporate the book atmosphere to set the appropriate mood and visual tone\n"
            "- Include the specified artistic style\n\n"
            
            "Image prompt:"
        )
        
        print(f'llm_prompt: \n{llm_prompt}')
        
        return llm_prompt
    
    
    def generate_image_prompt(self, character_name: str) -> str:
        """
        Generate an enhanced image prompt for a specific character.
        
        Args:
            character_name: Name of the character to generate a prompt for
            
        Returns:
            Enhanced image prompt string
        """
        # Load character data
        character_data = self._get_character_data(character_name)
        
        # Build the LLM prompt
        llm_prompt = self._build_llm_prompt(character_data, character_name)
        
        print("Generating Image prompt.....")
        
        # Generate the enhanced prompt
        image_prompt = self.llm.generate(
            llm_prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

        return image_prompt
        