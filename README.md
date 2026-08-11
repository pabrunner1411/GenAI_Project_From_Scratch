# GenAI Project From Scratch

## Part 1 — Gradio Semantic Chunking

### Installation

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Start

```powershell
& .\.venv\Scripts\python.exe .\app.py
```

Danach die im Terminal ausgegebene lokale URL im Browser öffnen.

Beim ersten Start lädt Sentence Transformers das ausgewählte Embedding-Modell
herunter. Danach wird es lokal aus dem Cache verwendet.

## Part 2 — feature_extraction

Extracts character profiles and overall mood/atmosphere from a book using spaCy NER + a Gemini LLM. Logic lives in [`feature_extraction/utils/`](feature_extraction/utils); [`feature_extraction/test.ipynb`](feature_extraction/test.ipynb) is a showcase notebook that runs the pipeline end to end.

Setup:
```
pip install -r feature_extraction/requirements.txt
python -m spacy download en_core_web_sm
```
Add a `GEMINI_API_KEY` to `feature_extraction/.env` (see `.env.example`), then run `test.ipynb`.
