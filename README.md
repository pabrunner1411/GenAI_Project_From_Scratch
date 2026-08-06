<<<<<<< HEAD
# template
=======
# Semantic Chunking + Character Feature Extraction

This project combines **Part 1** and **Part 2** in a single Gradio application.

- Part 1 loads TXT, Markdown, PDF, or DOCX files, cleans the text, and creates semantic chunks.
- Part 2 uses those chunks for spaCy name detection, Gemini character profiles, and book-atmosphere extraction.
- The previous sample image has been replaced in Gradio by a character-profile display with a dropdown selector.
- The original Part 2 structure under `feature_extraction/` is preserved.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For German documents, optionally install the German spaCy model:

```bash
python -m spacy download de_core_news_sm
```

## Gemini API key

Copy:

```text
feature_extraction/.env.example
```

to:

```text
feature_extraction/.env
```

Then enter your key:

```env
GEMINI_API_KEY=your_api_key
```

## Start the application

### Windows PowerShell

```powershell
& .\.venv\Scripts\python.exe .\app.py
```

### macOS or Linux

```bash
python app.py
```

## Usage

1. Upload a document and click **“1. Semantically chunk text.”**
2. Configure the feature-extraction parameters.
3. Click **“2. Extract character profiles.”**
4. Select a character from the dropdown. The profile appears directly below it.

Before a new extraction is run, the dropdown displays the bundled Dracula example profiles.

## Project structure

```text
app.py                         # Shared Gradio interface
pipeline.py                    # Part 1 pipeline and integration state
semantic_chunking.py           # Semantic chunking logic
feature_extraction/
  pipeline.py                  # Connection between Part 1 and Part 2
  test.ipynb                   # Original Part 2 notebook
  data/                        # Part 2 example data
  utils/                       # Original feature-extraction modules
```

## Privacy notice

spaCy processing runs locally. Selected text excerpts are sent to the Gemini API to generate character profiles and the book atmosphere.
>>>>>>> 4337b22 (Part 1 and 2 combined)
