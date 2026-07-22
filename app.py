from pathlib import Path

import gradio as gr

from config import DEFAULT_ENCODING, DEFAULT_MODEL, MODEL_OPTIONS
from pipeline import PipelineError, process_document


# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DRACULA_IMAGE = PROJECT_DIR / "images" / "dracula.jpg"


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
    Führt die Dokumentenverarbeitung aus und wandelt
    PipelineError in eine sichtbare Gradio-Fehlermeldung um.
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
            f"Ein unerwarteter Fehler ist aufgetreten: {error}"
        ) from error


# ---------------------------------------------------------
# Vordefiniertes Bild anzeigen
# ---------------------------------------------------------

def show_dracula_image() -> str:
    """
    Gibt den Pfad zum vordefinierten Dracula-Bild zurück.
    """

    if not DRACULA_IMAGE.exists():
        raise gr.Error(
            "Das Dracula-Bild wurde nicht gefunden.\n\n"
            f"Erwarteter Pfad: {DRACULA_IMAGE}"
        )

    return str(DRACULA_IMAGE)


# ---------------------------------------------------------
# Gradio-Oberfläche
# ---------------------------------------------------------

with gr.Blocks(
    title="Semantic Chunking für GenAI",
) as app:

    # -----------------------------------------------------
    # Upload und Einstellungen
    # -----------------------------------------------------

    with gr.Row():

        with gr.Column(scale=1):
            uploaded_file = gr.File(
                label="Dokument hochladen",
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
                value=300,
                step=50,
                label="Minimale Chunkgröße",
                info=(
                    "Ein semantischer Chunk wird normalerweise "
                    "nicht vor dieser Tokenzahl getrennt."
                ),
            )

            max_tokens = gr.Slider(
                minimum=300,
                maximum=1600,
                value=900,
                step=50,
                label="Maximale Chunkgröße",
                info=(
                    "Spätestens bei dieser Tokenzahl wird ein "
                    "neuer Chunk begonnen."
                ),
            )

            breakpoint_percentile = gr.Slider(
                minimum=5,
                maximum=40,
                value=20,
                step=1,
                label="Semantische Trennstellen (%)",
                info=(
                    "Ein höherer Wert erzeugt tendenziell mehr "
                    "und kleinere Chunks."
                ),
            )

            overlap_units = gr.Slider(
                minimum=0,
                maximum=3,
                value=1,
                step=1,
                label="Überlappende Einheiten",
                info=(
                    "Übernimmt Absätze oder Satzgruppen aus dem "
                    "vorherigen Chunk."
                ),
            )

            encoding_name = gr.Dropdown(
                choices=[
                    "cl100k_base",
                    "o200k_base",
                ],
                value=DEFAULT_ENCODING,
                label="Token-Encoding",
            )

            model_name = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=DEFAULT_MODEL,
                label="Embedding-Modell",
            )

            strip_gutenberg = gr.Checkbox(
                value=True,
                label=(
                    "Project-Gutenberg-Kopf und -Fuß entfernen"
                ),
            )

    # -----------------------------------------------------
    # Verarbeitung starten
    # -----------------------------------------------------

    process_button = gr.Button(
        "Text semantisch chunken",
        variant="primary",
    )

    status = gr.Markdown()

    preview = gr.Dataframe(
        headers=[
            "Chunk",
            "Tokens",
            "Trennungsgrund",
            "Vorschau",
        ],
        datatype=[
            "number",
            "number",
            "str",
            "str",
        ],
        interactive=False,
        label="Vorschau der ersten 20 Chunks",
    )

    with gr.Row():
        cleaned_download = gr.File(
            label="Bereinigten Text herunterladen"
        )

        jsonl_download = gr.File(
            label="Chunks als JSONL herunterladen"
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
        ],
    )

    # -----------------------------------------------------
    # Dracula-Bild
    # -----------------------------------------------------

    gr.Markdown(
        """
        ---

        ## Vordefiniertes Dracula-Bild
        """
    )

    show_image_button = gr.Button(
        "Dracula-Bild anzeigen"
    )

    image_output = gr.Image(
        value=None,
        label="Dracula",
        interactive=False,
        height=500,
    )

    show_image_button.click(
        fn=show_dracula_image,
        inputs=[],
        outputs=[image_output],
    )


# ---------------------------------------------------------
# App starten
# ---------------------------------------------------------

if __name__ == "__main__":
    print(f"Projektordner: {PROJECT_DIR}")
    print(f"Bildpfad: {DRACULA_IMAGE}")
    print(f"Bild vorhanden: {DRACULA_IMAGE.exists()}")

    app.queue().launch(
        max_file_size="30mb",
        inbrowser=True,
    )