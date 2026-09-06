# book_portrait_ai

Turns a book into character portraits. Four stages chained in one Gradio app, running entirely on local HuggingFace models, no external API calls at inference time.

1. Semantic chunking: upload a document, clean it, split it into semantically coherent chunks.
2. Character extraction: spaCy NER plus a local LLM extract character profiles and overall book atmosphere from those chunks.
3. Image prompt generation: turns each character profile into a descriptive text prompt. Each prompt also works on its own if pasted into an external image generator.
4. Image generation: renders each character's prompt into an actual portrait locally, using Stable Diffusion 1.5. This was added after the fact, see the "Image generation (Part 4)" section below for the VRAM handoff it required.

## Critical: do not commit without being asked

The user has explicitly said not to commit, ever, unless they ask. As of this writing the entire restructure and every feature built this session sit uncommitted in the working tree on the `merge-parts` branch. That is intentional, not an oversight. Do not run `git commit`, `git add` followed by a commit, or anything that changes history, unless the user asks for it in that turn.

## How to run it

From the repo root, as a module, not as a script:

```
python -m book_portrait_ai.app
```

`book_portrait_ai/app.py` uses relative imports (`from .semantic_chunking.pipeline import ...`), so running it directly with `python book_portrait_ai/app.py` fails with an import error. It has to be run with `-m` from the repo root so the package resolves.

## Project layout

```
book_portrait_ai/
  app.py                      Gradio entrypoint, ties all four parts together
  feature_extraction_bridge.py   adapts character_extraction for the app
  image_prompt_bridge.py         adapts image_prompt_builder for the app
  image_generation_bridge.py     adapts image_generation for the app
  gpu_utils.py                   reports/releases GPU memory, used to hand VRAM between parts
  semantic_chunking/          Part 1
  character_extraction/       Part 2
  image_prompt_builder/       Part 3
  image_generation/           Part 4

data/         sample book text, predefined image folder, sample JSON input
notebooks/    end_to_end_showcase.ipynb, mirrors the app's four stages cell by cell
```

This used to be three separate top level folders (`semantic_chunking/`, `feature_extraction/`, `image_prompt_builder/`), each written as if it were the only thing in the project, stitched together with `sys.path` hacks. It was restructured into one package with proper relative imports throughout. If you see references anywhere to the old paths, they are stale.

## Local models

Both Part 2 and Part 3 use the same local model, `Qwen/Qwen2.5-1.5B-Instruct`, loaded through `transformers`. This replaced Gemini (Part 2's original design) and `mistralai/Mistral-7B-Instruct-v0.1` (Part 3's original design). The swap happened because the user's GPU (RTX 3060 Laptop, 6GB VRAM) cannot fit a 7B model, and reusing one small model everywhere avoids downloading multiple multi gigabyte checkpoints.

Both `character_extraction/llm_client.py` and `image_prompt_builder/builder.py` check `torch.cuda.is_available()` and pass `device=` accordingly. A plain `pip install -r requirements.txt` pulls the CPU only build of `torch` by default, which makes everything much slower. See the README's Installation section for the command to reinstall a CUDA build.

`image_prompt_builder/builder.py`'s `ImagePromptBuilder` caches its `transformers` pipeline on the instance (`_get_generator`). It used to rebuild the pipeline from scratch on every single call, which made generating prompts for several characters far slower than it needed to be. If you see that pattern reappear anywhere, it is a real bug, not a style choice.

That exact pattern did reappear, in Part 2: `feature_extraction_bridge.py` called `get_client()` fresh on every run of `extract_characters_and_atmosphere`, and `get_client()` just returns `LocalLLMClient(model_name)` with no caching, so Part 2 was reloading the whole model from scratch every time. Fixed the same way as Part 3 was: `feature_extraction_bridge.py` now has its own `_get_client()` caching a module level singleton, same pattern as its pre-existing `_get_nlp()`.

## Character name detection: how title resolution works, and why the obvious approach failed

`character_extraction/character_names.py` has two layers on top of plain spaCy NER:

`_merge_name_variants` merges detected names that are word level variants of each other (e.g. "dracula" and "count dracula") into whichever variant was mentioned most. Purely structural, no hardcoded vocabulary.

`resolve_title_references` recovers characters who are mostly referred to by title rather than by name (e.g. Dracula being called "the Count" through most of the book, so plain NER badly undercounts him). It uses a short list of generic English honorifics (`GENERIC_TITLES`), not anything specific to this book.

The first design tried was: for a given title, find which already detected character's name shows up most often in the same chunks as that title, and attribute the title's mentions to them if one candidate clearly dominates. Tested against the real book, this failed outright: Dracula ranked ninth out of ten candidates by that measure. The reason is structural, not a bug: passages about "the Count" are narrated by or mention the other characters present (Mina, Van Helsing, Harker), not Dracula's own rare name, since his name is exactly what the title is substituting for. Nearby name statistics point away from the right answer for a central antagonist, not toward it.

The working design instead looks for a direct textual anchor: does the title ever appear glued straight to a candidate's name somewhere in the whole book (e.g. "Count Dracula")? If that anchor is unique to one character, every bare use of the title gets attributed to them. This is not circular the way the first approach was, and it generalizes across books without hardcoding anything book specific. On the real book this took Dracula from 31 detected mentions (rank 12, outside the default top 5, never profiled) to 219 (rank 3, correctly profiled).

If anyone proposes rebuilding this with a heuristic based on nearby mentions or names appearing together, point them at this note first. It was tried and it does not work for this exact reason.

`resolve_title_references` can also return a `name -> titles` map (`return_title_map=True`) alongside the mention counts, e.g. `{'dracula': ['count']}`. `feature_extraction_bridge.py` uses this both to widen `get_character_snippets`'s matching to title only passages, and to populate an `aliases` field on each profile plus an "Also known as" column in the app's character table, so a title only character is visibly explained rather than just silently counted.

## Snippet selection for character profiles: why book order picked the wrong chunks

`character_extraction/character_names.py`'s `get_character_snippets` used to hand `extract_character_profiles` each character's first N matching chunks in book order. That looked fine until Dracula and Jonathan's profiles came out nearly identical (both "elderly, anxious, frightened"). The cause: Dracula's name is literally the book's title, and Jonathan's is in the table of contents, so both characters' first two matching chunks were the same front matter (title page, preface), which has zero narrative content. Only their third snippet actually differed, and even that one was thin.

The fix ranks each character's matched chunks by mention density (how many times their name or a resolved title appears in that chunk) instead of book order, so `extract_character_profiles`'s `chunks[:max_chunks]` gets the richest passages about that character, not whichever matched first in the book.

While building that fix, a second bug turned up: title matching was a plain substring check (`"the count" in chunk`), so it matched inside unrelated words such as "the country". Both matching paths, the character's own name and any resolved title, now use word boundary regexes instead.

## Character profile extraction: a chunk being genuinely about a character is not enough

Even with dense, relevant chunks (the fix above), Van Helsing's and Arthur's profiles once came back describing Lucy instead. Their top ranked chunk was Lucy's staking scene, where Van Helsing and Arthur are both named and acting repeatedly ("Van Helsing opened his missal", "Arthur placed the point over the heart"), so the chunk correctly ranked as dense in their names. But the vivid physical and emotional description in that same passage, "sleeping peacefully", "a face of equalled sweetness and purity", belongs to Lucy's body in the coffin, not to either of them. The small model latched onto whichever description read as most vivid and attributed it to whichever name it was asked about.

The fix is in the prompt, not the chunk selection: `character_extraction/character_profiles.py`'s `SYSTEM_INSTRUCTION` now explicitly warns that excerpts often describe multiple characters in the same scene, and tells the model to extract details only about the named character, never borrowing another character's appearance, actions, or mood even when that other character is the more vivid, emotional focus of the passage. The user prompt also restates the target name as "the ONLY character to describe" right above the excerpts.

If a future profile looks suspiciously like a different character (shared physical_appearance strings are the tell), check whether the chunk feeding it is a shared scene like this one before assuming it is a chunk selection problem again.

## Image prompt generation: chat formatting, and shared versus character specific content

`image_prompt_builder/builder.py`'s `generate_image_prompt` used to send the whole prompt to the `transformers` pipeline as one raw string, unlike `character_extraction/llm_client.py`, which uses chat formatted messages (`system` plus `user` roles). Qwen2.5 1.5B Instruct is an instruct tuned model, so the raw string form skipped its chat template, and generation would sometimes drift into narrating its own output instead of writing the prompt itself (one run ended with the model writing "This image prompt combines all the specified elements..."). It also carried a leftover `pad_token_id=50256`, GPT-2's end of text token id, meaningless for Qwen's tokenizer. Both are fixed now, the pipeline gets a `messages` list, and the bogus `pad_token_id` is gone.

`image_prompt_bridge.py`'s `_build_character_data` used to silently drop two of the four fields `character_profiles.py` extracts, `signature_pose_or_expression` and `vibe_and_mood`, only forwarding `physical_appearance` and `personality_traits`. Both fields now reach the builder and appear in its prompt template.

Prompts for different characters used to read as near identical, because `DEFAULT_STYLE_PROMPT` (a fixed art style string appended to every character) and the extracted `book_atmosphere` (also shared across every character) both described lighting and mood, so the phrases reinforced each other and crowded out character specific detail. `DEFAULT_STYLE_PROMPT` now covers medium and technique words only, no lighting or mood language, and `_build_llm_prompt` states the shared book aesthetic first, ending on the character specific section immediately before generation, so the model's attention lands on what should actually distinguish this character.

## Image generation (Part 4): why the text models get unloaded first

Part 4 renders each character's Part 3 prompt into an actual portrait via `image_generation/generator.py`'s `ImageGenerator`, wrapping a `diffusers` `StableDiffusionPipeline` (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the maintained community mirror since the original `runwayml/stable-diffusion-v1-5` listing was superseded). This is a different model family from Parts 2 and 3's Qwen2.5-1.5B, chosen the same way, SD1.5 fits a 6GB laptop GPU comfortably where SDXL typically wants 8GB or more.

Parts 2 and 3 each keep their Qwen pipeline cached in GPU memory for the life of the app session (see above). That is fine on its own, but stacking a diffusion pipeline on top of both cached text models risks exceeding 6GB. So `LocalLLMClient` (`character_extraction/llm_client.py`) and `ImagePromptBuilder` (`image_prompt_builder/builder.py`) both gained an `unload()` method (drops the cached pipeline, `gc.collect()`, `torch.cuda.empty_cache()` if CUDA is available), exposed at the bridge level as `feature_extraction_bridge.unload_text_extraction_model()` and `image_prompt_bridge.unload_image_prompt_model()`. `app.py`'s `run_image_generation` calls both automatically before generating images, no separate button, the Gradio UI is meant to just work. `book_portrait_ai/gpu_utils.py` holds the shared `describe_gpu_memory()`/`release_gpu_memory()` helpers both unload methods use.

The user explicitly wants the unloading visible, not just automatic: `notebooks/end_to_end_showcase.ipynb` has its own separate cell before Part 4 that calls both unload functions directly and prints `describe_gpu_memory()` before and after, so the effect is visible rather than hidden inside the generation call. Keep that cell if editing the notebook, it is there on purpose, not a leftover.

`ImagePromptBuilder.unload()` only needs to clear `self._generator` (the real pipeline). `image_prompt_builder/llm_backend.py`'s `ImagePromptLLM._get_llm()`, which would load a second, separate copy of the model as a raw `AutoModelForCausalLM`/`AutoTokenizer`, is dead code, `builder.py` never calls it, it only reads `llm.model_name` as a plain string and builds its own `transformers.pipeline(...)` instead. Left alone on purpose, same as the other known dead code below, but worth knowing so nobody "fixes" `unload()` to chase a model that was never actually loaded.

## Known issues, left alone on purpose

`book_portrait_ai/image_prompt_builder/model.py` has an undefined `DocumentChunk` reference and raises `NameError` if imported. It is never imported by the working code path (`builder.py`'s `from model import CharacterProfile` line is commented out), so it does not affect anything, but do not "helpfully" fix it without being asked. Same for `data/images/dracula.jpg` not existing on disk. The user confirmed both are fine as they are.

## Jupyter kernel

None of the kernels already registered on this machine (`gemini_app`, `python3`, others) point at this project's `.venv`. A dedicated kernel was registered:

```
python -m ipykernel install --user --name book_portrait_ai --display-name "book_portrait_ai (.venv)"
```

Any notebook in this repo needs that kernel selected, otherwise imports fail or hit a different Python version with different available packages. `notebooks/end_to_end_showcase.ipynb`'s metadata already points at it.

## Communication style

The user does not want em dashes, semicolons, or hyphens used as punctuation in prose, meaning documentation, notebook markdown, and chat responses. Rephrase with periods, commas, or restructured sentences instead. This does not apply to code syntax, CLI flags, file or variable names, or text generated at runtime by an LLM call.
