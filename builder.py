# src/book_portrait_ai/image_prompt/builder.py

import json
from typing import Dict, Any, Optional
from llm_backend import ImagePromptLLM
# from model import CharacterProfile

class ImagePromptBuilder:
    """
    Builds image generation prompts from character JSON data.
    This is separate from the character extraction pipeline.
    """
    
    # Default artistic style prompt for consistency across all characters
    DEFAULT_STYLE_PROMPT = (
        "digital illustration, storybook art style, warm color palette, "
        "soft lighting, painterly texture, detailed background, "
        "expressive brushwork, atmospheric, cohesive composition"
    )
    
    
    def __init__(self, style_prompt: Optional[str] = None):
        """
        Initialize the prompt builder.
        
        Args:
            style_prompt: Optional custom style prompt for consistency.
                         If None, uses the default.
        """
        self.style_prompt = style_prompt or self.DEFAULT_STYLE_PROMPT
        self.llm = None
    
    
    def _get_llm(self):
        """Load the LLM instance."""
        if self.llm is None:
            self.llm = ImagePromptLLM()
        return self.llm
    
    
    def _parse_json_input(self, json_input: str) -> Dict[str, Any]:
        """
        Parse JSON input from string or file content.
        
        Args:
            json_input: JSON string or file content
            
        Returns:
            Parsed dictionary
        """
        try:
            with open(json_input, "r", encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_input}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {json_input}")
    
    
    def _build_character_prompt(self, character_data: Dict[str, Any]) -> str:
        """
        Build a descriptive image prompt from character JSON data.
        Creates the character-specific prompt.
        
        Expected JSON structure:
        {
            "character_name": "Name",
            "description": "Text description of character",
            "visual_traits": ["trait1", "trait2"],
            "personality_traits": ["trait1", "trait2"],
            "book_aesthetic": {"mood": "...", "setting": "..."} 
        }
        """
        book_aesthetic = character_data.get("book_aesthetic", {})
        
        character_prompt = (
            f"Character Name: {character_data.get('character_name', 'Unknown')}\n "
            f"Visual Traits: {', '.join(character_data.get('visual_traits', []))}\n "
            f"Personality: {', '.join(character_data.get('personality_traits', []))}\n "
            f"Description: {character_data.get('description', '')}\n "
            f"Book aesthetic: \n"
            f"Mood: {book_aesthetic.get('mood', '')}\n "
            f"Settings: {book_aesthetic.get('setting', '')}\n "
        )
        
        return character_prompt
    
    
    def _build_llm_prompt(self, character_data: Dict[str, Any]) -> str:
        """
        Build the prompt to send to the LLM for enhancement.
        Combines character prompt with style consistency.
        
        Args:
            character_data: Character JSON dictionary
            
        Returns:
            LLM prompt string
        """
        # Build the character-specific prompt
        character_prompt = self._build_character_prompt(character_data)
        
        # Combine with style prompt for consistency
        combined_prompt = f"{character_prompt} \nImage Artistic style: {self.style_prompt}"
        
        llm_prompt = (
                        "You are an expert prompt engineer for AI image generation. Your task is to create a single, "
                        "coherent, descriptive prompt for a text-to-image diffusion model.\n\n"
                        "Based on the following character information, create a detailed image generation prompt.\n"
                        
                        "CHARACTER INFORMATION:\n"
                        f"{combined_prompt}\n\n"
                        
                        "INSTRUCTIONS:\n"
                        "- Create ONLY the image prompt text, nothing else\n"
                        "- Do not include code, explanations, or extra text\n"
                        "- Focus on visual details, lighting, composition, and mood\n"
                        "- The prompt should be about 2-4 sentences\n"
                        "- Start with the character name and key visual features\n\n"
                        
                        "Image prompt:"
                    )

        print(f'llm_prompt: \n{llm_prompt}')
        
        return llm_prompt
    
    
    def generate_image_prompt(self, json_input: str) -> str:
        """
        Generate an enhanced image prompt from character JSON data.
        
        Args:
            json_input: JSON file path or dictionary containing character data
            
        Returns:
            Enhanced image prompt string
        """
        # Parse the input
        character_data = self._parse_json_input(json_input)
        
        # Build the LLM prompt
        llm_prompt = self._build_llm_prompt(character_data)
        
        # Get the LLM instance
        llm = self._get_llm()
        
        print("Generating Image prompt.....")
        
        # Generate the enhanced prompt
        from transformers import pipeline
        generator = pipeline('text-generation', model=llm.model_name)
        
        result = generator(
            llm_prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256,
            return_full_text=False,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Extract the generated response
        image_prompt = result[0]['generated_text'].strip()
        
        print(f"Generated Image Prompt:\n{image_prompt}")
    
        return image_prompt
    
    
    # def update_character_profile_inplace(
    #     self, 
    #     character_profile: CharacterProfile, 
    #     image_prompt: Optional[str] = None
    #     ) -> None:
    #     """
    #     Update a CharacterProfile in-place with the generated image prompt and model name.
        
    #     Args:
    #         character_profile: The existing CharacterProfile to update (modified in-place)
    #         image_prompt: Optional pre-generated image prompt.
    #     """
    #     # Get the LLM instance for model name
    #     llm = self._get_llm()
        
    #     # Modify the existing object in-place
    #     character_profile.prompt = image_prompt
    #     character_profile.model_name = llm.model_name
        
    #     print(f"Updated profile for: {character_profile.character_name}")
    #     print(f"Model: {character_profile.model_name}")
        