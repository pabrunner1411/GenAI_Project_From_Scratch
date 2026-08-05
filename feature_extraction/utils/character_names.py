import unicodedata
from collections import Counter

import spacy


def load_nlp(model_name='en_core_web_sm'):
    return spacy.load(model_name)


def clean_name(name):
    normalized = unicodedata.normalize('NFKD', name)
    cleaned = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    return cleaned.strip()


def count_character_names(text, nlp, threshold=1):
    doc = nlp(text)

    names = [clean_name(ent.text) for ent in doc.ents if ent.label_ == 'PERSON']
    names = [name.lower().replace('\n', ' ') for name in names]
    counts = Counter(names)

    filtered = {
        name: count for name, count in counts.items()
        if count >= threshold and name != ''
    }

    return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))


def get_character_snippets(chunks, character_names):
    snippets = {name: [] for name in character_names}

    for chunk in chunks:
        lower_chunk = chunk.lower()
        for name in snippets:
            if name in lower_chunk:
                snippets[name].append(chunk)

    return snippets
