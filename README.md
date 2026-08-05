# GenAI Project From Scratch

Combined part 2 and 3
- Refer to `test_p3.ipynb` for example
- Part 3 currently runs on HuggingFace Model, but eventually should use the same model (Gemini) as Part 2.

Add a `HUGGINGFACEHUB_API_TOKEN` to `feature_extraction/.env` (see `.env.example`), then run `test_p3.ipynb`.


## feature_extraction

Extracts character profiles and overall mood/atmosphere from a book using spaCy NER + a Gemini LLM. Logic lives in [`feature_extraction/utils/`](feature_extraction/utils); [`feature_extraction/test.ipynb`](feature_extraction/test.ipynb) is a showcase notebook that runs the pipeline end to end.

Setup:
```
pip install -r feature_extraction/requirements.txt
python -m spacy download en_core_web_sm
```
Add a `GEMINI_API_KEY` to `feature_extraction/.env` (see `.env.example`), then run `test.ipynb`.
