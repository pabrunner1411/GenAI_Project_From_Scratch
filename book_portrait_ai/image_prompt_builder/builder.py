# src/book_portrait_ai/image_prompt/builder.py

import json
from typing import Dict, Any, Optional
from .llm_backend import ImagePromptLLM
# from model import CharacterProfile

class ImagePromptBuilder:
    """
    Builds image generation prompts from character JSON data.
    This is separate from the character extraction pipeline.
    """
    
    # Default artistic style prompt for consistency across all characters.
    # Medium/technique only - mood and lighting come from the book aesthetic
    # instead, so the two don't duplicate each other and crowd out the
    # character's own details.
    DEFAULT_STYLE_PROMPT = (
        "digital illustration, storybook art style, painterly texture, "
        "expressive brushwork, cohesive composition"
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
        self._generator = None


    def _get_llm(self):
        """Load the LLM instance."""
        if self.llm is None:
            self.llm = ImagePromptLLM()
        return self.llm


    def _get_generator(self):
        """Load the text-generation pipeline once and reuse it across calls."""
        if self._generator is None:
            import torch
            from transformers import pipeline

            llm = self._get_llm()
            device = 0 if torch.cuda.is_available() else -1
            self._generator = pipeline('text-generation', model=llm.model_name, device=device)
        return self._generator


    def unload(self):
        """Releases the generation pipeline from memory, if one was loaded."""
        from .. import gpu_utils

        if self._generator is not None:
            model_name = self.llm.model_name if self.llm else "image prompt LLM"
            self._generator = None
            gpu_utils.release_gpu_memory()
            print(f'Unloaded {model_name} (image prompt generation).')


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
        Build the character-specific portion of the prompt from character JSON data.

        Expected JSON structure:
        {
            "character_name": "Name",
            "description": "Text description of character, or None",
            "visual_traits": ["trait1", "trait2"],
            "personality_traits": ["trait1", "trait2"],
            "signature_pose_or_expression": "Text description of a characteristic pose or expression, or None",
            "vibe_and_mood": "Text description of the character's own mood/vibe",
            "creature_type": "What kind of being they are, e.g. human, vampire, talking animal, or None",
            "attire": "What they are wearing, or None",
            "age": "How old they appear or are said to be, or None",
            "book_aesthetic": {"mood": "...", "setting": "..."}
        }

        description, signature_pose_or_expression, creature_type, attire,
        and age are left out of the prompt entirely when None, rather than
        written as "unspecified", so the model isn't given something that
        reads like a real detail to build on.
        """
        character_prompt = (
            f"Character Name: {character_data.get('character_name', 'Unknown')}\n "
            f"Personality: {', '.join(character_data.get('personality_traits', []))}\n "
            f"Character Mood: {character_data.get('vibe_and_mood', '')}\n "
        )

        description = character_data.get("description")
        if description:
            character_prompt += f"Description: {description}\n "

        pose = character_data.get("signature_pose_or_expression")
        if pose:
            character_prompt += f"Signature Pose/Expression: {pose}\n "

        visual_traits = character_data.get("visual_traits")
        if visual_traits:
            character_prompt += f"Visual Traits: {', '.join(visual_traits)}\n "

        creature_type = character_data.get("creature_type")
        if creature_type:
            character_prompt += f"Creature/Species: {creature_type}\n "

        attire = character_data.get("attire")
        if attire:
            character_prompt += f"Attire: {attire}\n "

        age = character_data.get("age")
        if age:
            character_prompt += f"Age: {age}\n "

        return character_prompt
    
    
    SYSTEM_INSTRUCTION = (
        "You are an expert prompt engineer for AI image generation. Your task is to create a "
        "single, coherent, descriptive prompt for a text-to-image diffusion model, based on "
        "character information the user gives you.\n"
        "- Respond with ONLY the image prompt text itself, nothing else\n"
        "- Do not include code, explanations, headers, or any commentary about the prompt\n"
        "- Focus on visual details, lighting, composition, and mood\n"
        "- The prompt should be about 2-4 sentences\n"
        "- Start with the character name and key visual features\n"
        "- The character information given to you may leave some visual details unspecified, "
        "for example it may not say what the character is wearing. An image needs every detail "
        "resolved, so where something is unspecified you may invent a plausible one, but it "
        "must stay consistent with the character's stated personality, mood, and the book "
        "aesthetic given to you. Do not invent a detail that clashes with them, for example "
        "do not dress a sad or menacing character in something cheerful or delicate"
    )

    def _build_llm_prompt(self, character_data: Dict[str, Any]) -> str:
        """
        Build the user-turn prompt to send to the LLM for enhancement.

        Book aesthetic and art style are stated first, as shared backdrop.
        Character-specific details come last, right before generation
        starts, so the model's attention lands on what should distinguish
        this character rather than on the material every character shares.

        Args:
            character_data: Character JSON dictionary

        Returns:
            User prompt string (paired with SYSTEM_INSTRUCTION as a chat message)
        """
        book_aesthetic = character_data.get("book_aesthetic", {})
        character_prompt = self._build_character_prompt(character_data)

        llm_prompt = (
            "Based on the following information, create a detailed image generation prompt.\n\n"
            "BOOK AESTHETIC (shared backdrop/atmosphere across all characters, not the main focus):\n"
            f"Mood: {book_aesthetic.get('mood', '')}\n "
            f"Settings: {book_aesthetic.get('setting', '')}\n\n"
            f"Image Artistic style: {self.style_prompt}\n\n"
            "CHARACTER INFORMATION (this character should be the main visual focus):\n"
            f"{character_prompt}"
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
        messages = [
            {'role': 'system', 'content': self.SYSTEM_INSTRUCTION},
            {'role': 'user', 'content': llm_prompt},
        ]

        print("Generating Image prompt.....")

        # Generate the enhanced prompt
        generator = self._get_generator()

        result = generator(
            messages,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
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
        