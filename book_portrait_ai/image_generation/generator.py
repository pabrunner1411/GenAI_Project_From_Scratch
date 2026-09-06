"""Renders a character portrait locally from an image prompt (Part 4).

Uses Stable Diffusion 1.5 via `diffusers`. The Qwen2.5-1.5B model used by
Parts 2 and 3 is small enough to share a 6GB laptop GPU with other things,
but a diffusion pipeline is not - see feature_extraction_bridge.py's and
image_prompt_bridge.py's unload_*() functions, which release those models
before this one gets loaded.
"""

from .. import gpu_utils

# A community fine-tune of SD1.5 (same architecture, same 6GB budget,
# just different trained weights). Leans painterly/concept-art rather
# than anime or photoreal. Swapped in after Lykon/dreamshaper-8 (too
# anime-like) and emilianJR/epiCRealism (too uncanny/photoreal) both
# missed this project's "storybook art style" goal.
DEFAULT_MODEL_NAME = "stablediffusionapi/deliberate-v2"

# Kept generic and book agnostic, same spirit as image_prompt_builder's
# DEFAULT_STYLE_PROMPT - steers away from common diffusion artifacts
# rather than adding anything book or character specific. Expanded past
# the basics to cover the failure modes SD1.5-family models are known
# for (hands, limbs, anatomy).
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, jpeg artifacts, blurry, distorted, "
    "ugly, duplicate, morbid, mutilated, out of frame, watermark, "
    "signature, text, extra fingers, mutated hands, poorly drawn hands, "
    "poorly drawn face, mutation, deformed, bad anatomy, bad proportions, "
    "extra limbs, missing arms, missing legs, extra arms, extra legs, "
    "fused fingers, too many fingers, long neck, cloned face, disfigured"
)


class ImageGenerator:
    """Wraps a Stable Diffusion pipeline, cached on the instance.

    CLIP (the text encoder Stable Diffusion 1.5 uses) has a hard limit of
    77 tokens. Part 3's generated prompts routinely run well past that
    (measured 87 and 113 tokens on real prompts), and diffusers silently
    truncates whatever doesn't fit. `compel` works around this by
    splitting a long prompt into 77-token chunks, encoding each chunk
    through CLIP separately, and concatenating the resulting embeddings,
    so nothing gets dropped. It stays on SD1.5, no bigger/heavier model
    needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None
        self._compel = None

    def _get_pipeline(self):
        """Loads the diffusion pipeline once and reuses it across calls."""
        if self._pipeline is None:
            import torch
            from diffusers import StableDiffusionPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    variant="fp16" if device == "cuda" else None,
                )
            except Exception:
                # Not every checkpoint publishes fp16-specific weight
                # files under that variant tag; fall back to the
                # default weights (still cast to fp16 above) if so.
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name, torch_dtype=dtype
                )
            pipeline = pipeline.to(device)
            # Trades a little speed for lower peak VRAM, needed to stay
            # inside a 6GB laptop GPU alongside everything else running.
            pipeline.enable_attention_slicing()

            self._pipeline = pipeline

        return self._pipeline

    def _get_compel(self):
        """Builds the long-prompt embedder once, tied to this pipeline's tokenizer/encoder."""
        if self._compel is None:
            from compel import Compel

            pipeline = self._get_pipeline()
            self._compel = Compel(
                tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder
            )

        return self._compel

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ):
        """Renders a single image from a text prompt. Returns a PIL.Image."""
        pipeline = self._get_pipeline()
        compel = self._get_compel()

        prompt_embeds = compel(prompt)
        negative_prompt_embeds = compel(negative_prompt)
        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )

        result = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return result.images[0]

    def unload(self):
        """Releases the diffusion pipeline from memory, if one was loaded."""
        if self._pipeline is not None:
            self._pipeline = None
            self._compel = None
            gpu_utils.release_gpu_memory()
            print(f'Unloaded {self.model_name} (image generation).')
