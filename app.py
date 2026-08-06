from pathlib import Path
from typing import Any

import gradio as gr

from config import DEFAULT_ENCODING, DEFAULT_MODEL, MODEL_OPTIONS
from feature_extraction.pipeline import (
    FeatureExtractionError,
    extract_document_features,
    load_example_data,
)
from feature_extraction.utils.llm_client import DEFAULT_MODEL as GEMINI_MODEL
from pipeline import PipelineError, process_document_with_state


PROJECT_DIR = Path(__file__).resolve().parent
EXAMPLE_PROFILES, EXAMPLE_ATMOSPHERE = load_example_data()


def display_character_name(name: str) -> str:
    """Format an internal lowercase character name for the interface."""
    return name.replace("_", " ").title()


def character_choices(profiles: dict[str, Any]) -> list[tuple[str, str]]:
    """Create dropdown entries as (display name, internal name) pairs."""
    return [
        (display_character_name(name), name)
        for name in profiles
    ]


def preferred_character(profiles: dict[str, Any]) -> str | None:
    """Select Dracula as the example, otherwise the first available character."""
    if "dracula" in profiles:
        return "dracula"
    return next(iter(profiles), None)


def render_character_profile(
    selected_character: str | None,
    profiles: dict[str, Any] | None,
) -> str:
    """Render a structured character profile as Markdown."""
    if not selected_character or not profiles:
        return "*No character profile is available yet.*"

    profile = profiles.get(selected_character)
    if not profile:
        return "*No profile was found for this character.*"

    traits = profile.get("personality_traits", [])
    if isinstance(traits, str):
        traits = [traits]

    trait_text = ", ".join(str(trait) for trait in traits) or "–"

    return (
        f"### {display_character_name(selected_character)}\n\n"
        f"**Physical appearance**  \n"
        f"{profile.get('physical_appearance', '–')}\n\n"
        f"**Personality traits**  \n"
        f"{trait_text}\n\n"
        f"**Signature pose or expression**  \n"
        f"{profile.get('signature_pose_or_expression', '–')}\n\n"
        f"**Vibe and mood**  \n"
        f"{profile.get('vibe_and_mood', '–')}"
    )


def render_book_atmosphere(atmosphere: dict[str, Any] | None) -> str:
    """Render the book atmosphere extracted in Part 2."""
    if not atmosphere:
        return "*No book atmosphere is available yet.*"

    if "error" in atmosphere:
        return f"⚠️ {atmosphere['error']}"

    keywords = atmosphere.get("overall_mood_keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    keyword_text = ", ".join(str(item) for item in keywords) or "–"

    return (
        f"**Mood keywords:** {keyword_text}\n\n"
        f"**Visual palette and lighting:**  \n"
        f"{atmosphere.get('visual_palette_and_lighting', '–')}\n\n"
        f"**Narrative tone progression:**  \n"
        f"{atmosphere.get('narrative_tone_progression', '–')}"
    )


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
    """Run Part 1 and store its output paths for Part 2."""
    try:
        return process_document_with_state(
            uploaded_file=uploaded_file,
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
            breakpoint_percentile=float(breakpoint_percentile),
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


def run_feature_extraction(
    processing_state: dict[str, str] | None,
    name_threshold: int,
    max_characters: int,
    max_chunks_per_character: int,
    nlp_model: str,
    gemini_model: str,
):
    """Run Part 2 and update the dropdown and profile display."""
    try:
        (
            feature_status,
            profiles,
            atmosphere,
            profiles_path,
            atmosphere_path,
        ) = extract_document_features(
            processing_state=processing_state,
            name_threshold=int(name_threshold),
            max_characters=int(max_characters),
            max_chunks_per_character=int(max_chunks_per_character),
            nlp_model=nlp_model,
            gemini_model=gemini_model,
        )
    except FeatureExtractionError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:
        raise gr.Error(
            f"Feature extraction failed: {error}"
        ) from error

    selected = preferred_character(profiles)

    return (
        feature_status,
        gr.Dropdown(
            choices=character_choices(profiles),
            value=selected,
        ),
        profiles,
        render_character_profile(selected, profiles),
        render_book_atmosphere(atmosphere),
        profiles_path,
        atmosphere_path,
    )


INITIAL_CHARACTER = preferred_character(EXAMPLE_PROFILES)

with gr.Blocks(
    title="Semantic Chunking and Character Feature Extraction",
) as app:
    processing_state = gr.State(value=None)
    profiles_state = gr.State(value=EXAMPLE_PROFILES)

    gr.Markdown(
        """
        # Semantic Chunking and Character Profiles

        **Part 1** semantically splits the uploaded document. **Part 2** then
        uses those chunks to detect characters and create structured profiles
        with Gemini.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_file = gr.File(
                label="Upload document",
                file_types=[".txt", ".md", ".pdf", ".docx"],
                type="filepath",
            )

        with gr.Column(scale=1):
            min_tokens = gr.Slider(
                minimum=100,
                maximum=800,
                value=300,
                step=50,
                label="Minimum chunk size",
                info=(
                    "A semantic chunk is normally not split before it reaches "
                    "this token count."
                ),
            )
            max_tokens = gr.Slider(
                minimum=300,
                maximum=1600,
                value=900,
                step=50,
                label="Maximum chunk size",
                info=(
                    "A new chunk is started no later than this token count."
                ),
            )
            breakpoint_percentile = gr.Slider(
                minimum=5,
                maximum=40,
                value=20,
                step=1,
                label="Semantic breakpoints (%)",
                info=(
                    "A higher value generally produces more, smaller chunks."
                ),
            )
            overlap_units = gr.Slider(
                minimum=0,
                maximum=3,
                value=1,
                step=1,
                label="Overlapping units",
                info=(
                    "Copies paragraphs or sentence groups from the previous "
                    "chunk."
                ),
            )
            encoding_name = gr.Dropdown(
                choices=["cl100k_base", "o200k_base"],
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
                label="Remove Project Gutenberg header and footer",
            )

    process_button = gr.Button(
        "1. Semantically chunk text",
        variant="primary",
    )
    status = gr.Markdown()
    preview = gr.Dataframe(
        headers=["Chunk", "Tokens", "Break reason", "Preview"],
        datatype=["number", "number", "str", "str"],
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
            processing_state,
        ],
    )

    gr.Markdown(
        """
        ---

        ## Character Profiles

        The previous Dracula sample image has been replaced with the profile
        display. The dropdown initially contains the bundled Dracula example
        profiles. For an uploaded document, run Part 1 first and then start
        feature extraction.

        **Privacy notice:** Selected text excerpts are sent to the Gemini API
        for analysis.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            name_threshold = gr.Slider(
                minimum=1,
                maximum=30,
                value=5,
                step=1,
                label="Minimum number of name mentions",
            )
            max_characters = gr.Slider(
                minimum=1,
                maximum=30,
                value=12,
                step=1,
                label="Maximum number of characters to analyze",
                info="Each character triggers a separate Gemini request.",
            )
            max_chunks_per_character = gr.Slider(
                minimum=1,
                maximum=8,
                value=3,
                step=1,
                label="Text excerpts per character",
            )
            nlp_model = gr.Dropdown(
                choices=["en_core_web_sm", "de_core_news_sm"],
                value="en_core_web_sm",
                allow_custom_value=True,
                label="spaCy language model",
            )
            gemini_model = gr.Textbox(
                value=GEMINI_MODEL,
                label="Gemini model",
            )
            extract_button = gr.Button(
                "2. Extract character profiles",
                variant="primary",
            )

        with gr.Column(scale=2):
            feature_status = gr.Markdown(
                "*The bundled Dracula profiles are currently displayed.*"
            )
            character_dropdown = gr.Dropdown(
                choices=character_choices(EXAMPLE_PROFILES),
                value=INITIAL_CHARACTER,
                label="Select a character",
                filterable=True,
            )
            profile_output = gr.Markdown(
                render_character_profile(
                    INITIAL_CHARACTER,
                    EXAMPLE_PROFILES,
                )
            )

    with gr.Accordion("Book atmosphere", open=False):
        atmosphere_output = gr.Markdown(
            render_book_atmosphere(EXAMPLE_ATMOSPHERE)
        )

    with gr.Row():
        profiles_download = gr.File(
            label="Download character profiles as JSON"
        )
        atmosphere_download = gr.File(
            label="Download book atmosphere as JSON"
        )

    character_dropdown.change(
        fn=render_character_profile,
        inputs=[character_dropdown, profiles_state],
        outputs=[profile_output],
    )

    extract_button.click(
        fn=run_feature_extraction,
        inputs=[
            processing_state,
            name_threshold,
            max_characters,
            max_chunks_per_character,
            nlp_model,
            gemini_model,
        ],
        outputs=[
            feature_status,
            character_dropdown,
            profiles_state,
            profile_output,
            atmosphere_output,
            profiles_download,
            atmosphere_download,
        ],
    )


if __name__ == "__main__":
    print(f"Project directory: {PROJECT_DIR}")
    app.queue().launch(
        max_file_size="30mb",
        inbrowser=True,
    )
