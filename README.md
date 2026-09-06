# GenAI Project From Scratch

Turns a book into character portraits. It semantically chunks a document, extracts character profiles and book atmosphere, generates an image prompt per character, then renders the actual portraits, all chained together in one Gradio app that runs entirely on local HuggingFace models.

## Project structure

```
book_portrait_ai/
  app.py                      # Gradio entrypoint, ties all four parts together
  feature_extraction_bridge.py
  image_prompt_bridge.py
  image_generation_bridge.py
  gpu_utils.py                 # reporting/releasing GPU memory, used to hand off VRAM between parts
  semantic_chunking/          # Part 1: document loading, cleaning, semantic chunking
  character_extraction/       # Part 2: spaCy NER + local LLM character/atmosphere extraction
  image_prompt_builder/       # Part 3: turns a character profile into an image prompt
  image_generation/           # Part 4: renders an image per character locally

data/                         # sample book text, predefined image, sample JSON input
notebooks/
  end_to_end_showcase.ipynb   # runs the same pipeline as the app, cell by cell
```

## Installation

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

If you have an NVIDIA GPU, note that the `pip install` above pulls in a CPU
only build of `torch` by default, which makes the local HuggingFace models
(see below) much slower than necessary. To use the GPU instead, reinstall `torch` from
PyTorch's CUDA index after the step above (pick the `cuXXX` tag matching your
driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```powershell
& .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps torch
```

## Running the app

From the repo root (the app uses relative imports, so it must be run as a module, not as a script):

```powershell
& .\.venv\Scripts\python.exe -m book_portrait_ai.app
```

Then open the local URL printed in the terminal in your browser.

## Part 1: Semantic Chunking

Uploads a document, cleans it, and splits it into semantically coherent chunks. Logic lives in [`book_portrait_ai/semantic_chunking/`](book_portrait_ai/semantic_chunking).

On first start, Sentence Transformers downloads the selected embedding
model. After that, it's used locally from the cache.

## Part 2: Character Extraction

Extracts character profiles and overall mood/atmosphere from a book using spaCy NER + a local HuggingFace LLM (`Qwen/Qwen2.5-1.5B-Instruct`, downloaded and run locally on your machine without needing an API key). Logic lives in [`book_portrait_ai/character_extraction/`](book_portrait_ai/character_extraction).

Optionally add a `HUGGINGFACEHUB_API_TOKEN` to `.env` (see `.env.example`) if you want to use a gated/private model or raise HF download rate limits. From the app, after running "Chunk text semantically", use the "Part 2: Extract characters & atmosphere" section below it. This reuses Part 1's semantic chunks instead of chunking the text again. Wired up via [`book_portrait_ai/feature_extraction_bridge.py`](book_portrait_ai/feature_extraction_bridge.py).

## Part 3: Image Prompt Builder

Turns character JSON data into a refined, descriptive **text prompt** via [`book_portrait_ai/image_prompt_builder/builder.py`](book_portrait_ai/image_prompt_builder/builder.py)'s `ImagePromptBuilder`, using the same local HuggingFace LLM as Part 2 (`Qwen/Qwen2.5-1.5B-Instruct`, defined in [`book_portrait_ai/image_prompt_builder/llm_backend.py`](book_portrait_ai/image_prompt_builder/llm_backend.py)). Part 4 renders these prompts into images, but each prompt also works on its own if pasted into an external image generator.

From the app, after running "Extract characters & atmosphere", use the "Part 3: Generate image prompts" section below it. This generates one prompt per profiled character, using their profile and the book's atmosphere. Wired up via [`book_portrait_ai/image_prompt_bridge.py`](book_portrait_ai/image_prompt_bridge.py).

## Part 4: Image Generation

Renders one portrait per profiled character locally, using Stable Diffusion 1.5 via `diffusers`, through [`book_portrait_ai/image_generation/generator.py`](book_portrait_ai/image_generation/generator.py)'s `ImageGenerator`. Wired up via [`book_portrait_ai/image_generation_bridge.py`](book_portrait_ai/image_generation_bridge.py).

From the app, after running "Generate image prompts", use the "Part 4: Generate images" section below it.

Part 3's prompts routinely run past CLIP's hard 77 token limit (measured over 110 tokens on real prompts), which `diffusers` would otherwise silently truncate. `ImageGenerator` uses the `compel` library to chunk long prompts and encode them through CLIP without dropping anything.

Stable Diffusion needs its own GPU memory, so on a 6GB laptop GPU (the same constraint that led to `Qwen2.5-1.5B-Instruct` over a 7B model for Parts 2 and 3) it may not fit alongside the already-loaded text models, and vice versa. The handoff runs both ways: `run_image_generation` in `app.py` releases the Part 2/3 text models before generating images, and `run_feature_extraction`/`run_image_prompt_generation` release the Part 4 image model before loading their own, so re-running any part after another just works rather than risking two models fighting over the same 6GB. [`book_portrait_ai/gpu_utils.py`](book_portrait_ai/gpu_utils.py) reports and releases GPU memory, and `unload_text_extraction_model()`/`unload_image_prompt_model()`/`unload_image_generation_model()` (in `feature_extraction_bridge.py`/`image_prompt_bridge.py`/`image_generation_bridge.py`) do the actual releasing.

## Notebook

[`notebooks/end_to_end_showcase.ipynb`](notebooks/end_to_end_showcase.ipynb) runs the exact same pipeline as the app (`process_document`, `extract_characters_and_atmosphere`, `generate_image_prompts`, `generate_character_images`) cell by cell against [`data/dracula.txt`](data/dracula.txt), for testing or showcasing the whole flow without the UI. It also includes its own explicit "free GPU memory" cell before image generation, so you can see what gets released rather than have it happen silently. It uses the `book_portrait_ai (.venv)` Jupyter kernel (registered via `python -m ipykernel install --user --name book_portrait_ai --display-name "book_portrait_ai (.venv)"` from the project's `.venv`). Make sure that kernel is selected when opening it, otherwise imports will fail against whatever other Python environment Jupyter defaults to.
