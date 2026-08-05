import os
from huggingface_hub import login
from transformers import pipeline


class ImagePromptLLM:
    """
    Uses an LLM to enhance/refine the image prompt from JSON data.
    This uses the model via HuggingFace API.
    """
    def __init__(self, model_name=None):
        """
        Initialize with HuggingFace model.
        """
        self.model_name = model_name
        self.token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._model = None
        
        if self.token:
            login(token=self.token)
        
    
    def load(self):
        """
        Load the LLM.
        """
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.token
                    )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=self.token
                    )
        
        print(f'LLM model {self.model_name} activated!')
        
        return self
    
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the model.
        """
        generator = pipeline(
            'text-generation',
            model=self._model,
            tokenizer=self._tokenizer,
            device=0
        )
        
        # Default parameters
        params = {
            'max_new_tokens': kwargs.get('max_new_tokens', 500),
            'temperature': kwargs.get('temperature', 0.7),
            'do_sample': kwargs.get('do_sample', True),
            'pad_token_id': self._tokenizer.eos_token_id,
            'return_full_text': kwargs.get('return_full_text', False),
            'top_p': kwargs.get('top_p', 0.9),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.1)
        }
        
        result = generator(prompt, **params)
        return result[0]['generated_text'].strip()