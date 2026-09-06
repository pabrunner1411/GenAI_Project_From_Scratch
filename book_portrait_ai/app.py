from pathlib import Path

import gradio as gr

from .feature_extraction_bridge import (
    FeatureExtractionError,
    extract_characters_and_atmosphere,
    unload_text_extraction_model,
)
from .image_generation_bridge import (
    ImageGenerationError,
    generate_character_images,
    unload_image_generation_model,
)
from .image_prompt_bridge import (
    ImagePromptGenerationError,
    generate_image_prompts,
    unload_image_prompt_model,
)
from .semantic_chunking.config import DEFAULT_ENCODING, DEFAULT_MODEL, MODEL_OPTIONS
from .semantic_chunking.pipeline import PipelineError, process_document


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

def run_pipeline(
    uploaded_file: str | None,
    min_tokens: int,
    max_tokens: int,
    breakpoint_percentile: float,
    overlap_units: int,
    encoding_name: str,
    model_name: str,
    strip_gutenberg: bool,
):
    """
    Runs document processing and converts
    PipelineError into a visible Gradio error message.
    """

    try:
        return process_document(
            uploaded_file=uploaded_file,
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
            breakpoint_percentile=float(
                breakpoint_percentile
            ),
            overlap_units=int(overlap_units),
            encoding_name=encoding_name,
            model_name=model_name,
            strip_gutenberg=strip_gutenberg,
        )

    except PipelineError as error:
        raise gr.Error(str(error)) from error

    except Exception as error:
        raise gr.Error(
            f"An unexpected error occurred: {error}"
        ) from error


# ---------------------------------------------------------
# Character & atmosphere extraction (Part 2)
# ---------------------------------------------------------

def run_feature_extraction(
    cleaned_text: str | None,
    chunks: list | None,
    name_threshold: int,
    max_characters: int,
    max_chunks_per_character: int,
):
    """
    Runs character and atmosphere extraction and converts
    FeatureExtractionError into a visible Gradio error message.

    Releases the Part 4 image model first, since it may still be loaded
    from an earlier run and Part 2's own text model needs GPU memory too.
    """

    try:
        unload_image_generation_model()
        status, name_rows, profiles, atmosphere = extract_characters_and_atmosphere(
            cleaned_text=cleaned_text,
            chunks=chunks,
            name_threshold=int(name_threshold),
            max_characters=int(max_characters),
            max_chunks_per_character=int(max_chunks_per_character),
        )
        # profiles/atmosphere are returned twice: once for the visible JSON
        # outputs, once to populate the hidden state Part 3 reads from.
        return status, name_rows, profiles, atmosphere, profiles, atmosphere

    except FeatureExtractionError as error:
        raise gr.Error(str(error)) from error

    except Exception as error:
        raise gr.Error(
            f"An unexpected error occurred: {error}"
        ) from error


# ---------------------------------------------------------
# Image prompt generation (Part 3)
# ---------------------------------------------------------

def run_image_prompt_generation(
    character_profiles: dict | None,
    book_atmosphere: dict | None,
):
    """
    Runs image prompt generation and converts
    ImagePromptGenerationError into a visible Gradio error message.

    Releases the Part 4 image model first, in case Part 3 is re-run on
    its own after Part 4 already loaded Stable Diffusion.
    """

    try:
        unload_image_generation_model()
        status, prompts = generate_image_prompts(character_profiles, book_atmosphere)
        # prompts is returned twice: once for the visible JSON output,
        # once to populate the hidden state Part 4 reads from.
        return status, prompts, prompts

    except ImagePromptGenerationError as error:
        raise gr.Error(str(error)) from error

    except Exception as error:
        raise gr.Error(
            f"An unexpected error occurred: {error}"
        ) from error


# ---------------------------------------------------------
# Image generation (Part 4)
# ---------------------------------------------------------

def run_image_generation(image_prompts: dict | None):
    """
    Runs local image generation and converts
    ImageGenerationError into a visible Gradio error message.

    Releases the Part 2/3 text models first, since Stable Diffusion
    needs GPU memory they would otherwise still be holding.
    """

    try:
        unload_text_extraction_model()
        unload_image_prompt_model()
        return generate_character_images(image_prompts)

    except ImageGenerationError as error:
        raise gr.Error(str(error)) from error

    except Exception as error:
        raise gr.Error(
            f"An unexpected error occurred: {error}"
        ) from error


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------

with gr.Blocks(
    title="Semantic Chunking for GenAI",
) as app:

    # -----------------------------------------------------
    # Upload and settings
    # -----------------------------------------------------

    with gr.Row():

        with gr.Column(scale=1):
            uploaded_file = gr.File(
                label="Upload document",
                file_types=[
                    ".txt",
                    ".md",
                    ".pdf",
                    ".docx",
                ],
                type="filepath",
            )

        with gr.Column(scale=1):
            min_tokens = gr.Slider(
                minimum=100,
                maximum=800,
                value=200,
                step=50,
                label="Minimum chunk size",
                info=(
                    "A semantic chunk is normally not split "
                    "before reaching this token count."
                ),
            )

            max_tokens = gr.Slider(
                minimum=300,
                maximum=3500,
                value=1000,
                step=50,
                label="Maximum chunk size",
                info=(
                    "A new chunk is started at the latest "
                    "once this token count is reached."
                ),
            )

            breakpoint_percentile = gr.Slider(
                minimum=5,
                maximum=40,
                value=20,
                step=1,
                label="Semantic breakpoints (%)",
                info=(
                    "A higher value tends to produce more, "
                    "smaller chunks."
                ),
            )

            overlap_units = gr.Slider(
                minimum=0,
                maximum=3,
                value=1,
                step=1,
                label="Overlapping units",
                info=(
                    "Carries paragraphs or sentence groups over "
                    "from the previous chunk."
                ),
            )

            encoding_name = gr.Dropdown(
                choices=[
                    "cl100k_base",
                    "o200k_base",
                ],
                value=DEFAULT_ENCODING,
                label="Token encoding",
            )

            model_name = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=DEFAULT_MODEL,
                label="Embedding model",
            )

            strip_gutenberg = gr.Checkbox(
                value=True,
                label=(
                    "Remove Project Gutenberg header and footer"
                ),
            )

    # -----------------------------------------------------
    # Run everything at once
    # -----------------------------------------------------

    run_all_button = gr.Button(
        "Run all 4 steps",
        variant="primary",
    )
    gr.Markdown(
        "Runs Parts 1-4 back to back using the settings above and below, "
        "so you don't have to click through and wait at each step "
        "individually. Stops and shows an error if any step fails, "
        "rather than continuing with bad input.\n\n---"
    )

    # -----------------------------------------------------
    # Start processing
    # -----------------------------------------------------

    process_button = gr.Button(
        "Chunk text semantically",
        variant="primary",
    )

    status = gr.Markdown()

    preview = gr.Dataframe(
        headers=[
            "Chunk",
            "Tokens",
            "Break reason",
            "Preview",
        ],
        datatype=[
            "number",
            "number",
            "str",
            "str",
        ],
        interactive=False,
        label="Preview of the first 20 chunks",
    )

    with gr.Row():
        cleaned_download = gr.File(
            label="Download cleaned text"
        )

        jsonl_download = gr.File(
            label="Download chunks as JSONL"
        )

    cleaned_text_state = gr.State()
    chunks_state = gr.State()

    process_button.click(
        fn=run_pipeline,
        inputs=[
            uploaded_file,
            min_tokens,
            max_tokens,
            breakpoint_percentile,
            overlap_units,
            encoding_name,
            model_name,
            strip_gutenberg,
        ],
        outputs=[
            status,
            preview,
            cleaned_download,
            jsonl_download,
            cleaned_text_state,
            chunks_state,
        ],
    )

    # -----------------------------------------------------
    # Part 2: Extract characters & atmosphere
    # -----------------------------------------------------

    gr.Markdown(
        """
        ---

        ## Part 2: Extract characters & atmosphere
        """
    )

    with gr.Row():
        name_threshold = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Minimum number of name mentions",
            info=(
                "Only characters mentioned at least this "
                "often are considered relevant."
            ),
        )

        max_characters = gr.Slider(
            minimum=1,
            maximum=15,
            value=5,
            step=1,
            label="Maximum number of profiled characters",
            info=(
                "Limits the number of LLM calls "
                "(cost/duration)."
            ),
        )

        max_chunks_per_character = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Chunks per character profile",
            info=(
                "How many of a character's matched chunks "
                "(most mentions first) feed their profile. "
                "More context, but a longer prompt for the "
                "LLM to reconcile."
            ),
        )

    extract_button = gr.Button(
        "Extract characters & atmosphere",
        variant="primary",
    )

    feature_status = gr.Markdown()

    character_counts_table = gr.Dataframe(
        headers=["Character", "Mentions", "Also known as"],
        datatype=["str", "number", "str"],
        interactive=False,
        label="Detected characters",
    )

    with gr.Row():
        character_profiles_output = gr.JSON(
            label="Character profiles"
        )

        book_atmosphere_output = gr.JSON(
            label="Book atmosphere"
        )

    character_profiles_state = gr.State()
    book_atmosphere_state = gr.State()

    extract_button.click(
        fn=run_feature_extraction,
        inputs=[
            cleaned_text_state,
            chunks_state,
            name_threshold,
            max_characters,
            max_chunks_per_character,
        ],
        outputs=[
            feature_status,
            character_counts_table,
            character_profiles_output,
            book_atmosphere_output,
            character_profiles_state,
            book_atmosphere_state,
        ],
    )

    # -----------------------------------------------------
    # Part 3: Generate image prompts
    # -----------------------------------------------------

    gr.Markdown(
        """
        ---

        ## Part 3: Generate image prompts
        """
    )

    generate_prompts_button = gr.Button(
        "Generate image prompts",
        variant="primary",
    )

    image_prompts_status = gr.Markdown()

    image_prompts_output = gr.JSON(
        label="Image prompts"
    )

    image_prompts_state = gr.State()

    generate_prompts_button.click(
        fn=run_image_prompt_generation,
        inputs=[
            character_profiles_state,
            book_atmosphere_state,
        ],
        outputs=[
            image_prompts_status,
            image_prompts_output,
            image_prompts_state,
        ],
    )

    # -----------------------------------------------------
    # Part 4: Generate images
    # -----------------------------------------------------

    gr.Markdown(
        """
        ---

        ## Part 4: Generate images
        """
    )

    generate_images_button = gr.Button(
        "Generate images",
        variant="primary",
    )

    image_generation_status = gr.Markdown()

    character_images_output = gr.Gallery(
        label="Character portraits"
    )

    generate_images_button.click(
        fn=run_image_generation,
        inputs=[
            image_prompts_state,
        ],
        outputs=[
            image_generation_status,
            character_images_output,
        ],
    )

    # -----------------------------------------------------
    # Run all 4 steps: chains the same handlers as the
    # individual buttons above, one after another. .then()
    # only runs once the previous step finishes, and stops
    # the chain if a step raises (e.g. a bad upload), instead
    # of running later steps on incomplete data.
    # -----------------------------------------------------

    run_all_button.click(
        fn=run_pipeline,
        inputs=[
            uploaded_file,
            min_tokens,
            max_tokens,
            breakpoint_percentile,
            overlap_units,
            encoding_name,
            model_name,
            strip_gutenberg,
        ],
        outputs=[
            status,
            preview,
            cleaned_download,
            jsonl_download,
            cleaned_text_state,
            chunks_state,
        ],
    ).then(
        fn=run_feature_extraction,
        inputs=[
            cleaned_text_state,
            chunks_state,
            name_threshold,
            max_characters,
            max_chunks_per_character,
        ],
        outputs=[
            feature_status,
            character_counts_table,
            character_profiles_output,
            book_atmosphere_output,
            character_profiles_state,
            book_atmosphere_state,
        ],
    ).then(
        fn=run_image_prompt_generation,
        inputs=[
            character_profiles_state,
            book_atmosphere_state,
        ],
        outputs=[
            image_prompts_status,
            image_prompts_output,
            image_prompts_state,
        ],
    ).then(
        fn=run_image_generation,
        inputs=[
            image_prompts_state,
        ],
        outputs=[
            image_generation_status,
            character_images_output,
        ],
    )


# ---------------------------------------------------------
# Start app
# ---------------------------------------------------------

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")

    app.queue().launch(
        max_file_size="30mb",
        inbrowser=True,
    )