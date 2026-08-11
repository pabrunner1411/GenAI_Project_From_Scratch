import os
from huggingface_hub import login


class ImagePromptLLM:
    """
    Uses an LLM to enhance/refine the image prompt from JSON data.
    This uses the model via HuggingFace API.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize with HuggingFace model.
        """
        self.model_name = model_name
        self.token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._llm = None
        
        if self.token:
            login(token=self.token)
        
    
    def _get_llm(self):
        """Load the LLM."""
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.token
                    )
            self._llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=self.token
                    )
        
        print(f'LLM model {self.model_name} activated!')
        
        return self._llm