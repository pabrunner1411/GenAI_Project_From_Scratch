import re
import unicodedata
from collections import Counter

import spacy

# Small set of common English honorifics, not specific to any one book.
# Used to recover characters who are mostly referred to by title rather
# than by name (e.g. "the Count" instead of "Dracula").
GENERIC_TITLES = [
    'count', 'countess', 'king', 'queen', 'doctor', 'professor',
    'captain', 'colonel', 'general', 'father', 'sir', 'lady', 'lord',
    'baron', 'duke', 'duchess', 'mr', 'mrs', 'miss', 'madame',
]


def load_nlp(model_name='en_core_web_sm'):
    return spacy.load(model_name)


def clean_name(name):
    normalized = unicodedata.normalize('NFKD', name)
    cleaned = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    return cleaned.strip()


def _normalize_word(word):
    """Collapses simple plural/possessive forms (e.g. 'draculas' -> 'dracula')."""
    return word[:-1] if word.endswith('s') and len(word) > 3 else word


def _is_word_subsequence(short, long_):
    """True if `short` (a tuple of words) is a contiguous run within `long_`."""
    span = len(short)
    return any(
        long_[i:i + span] == short
        for i in range(len(long_) - span + 1)
    )


def _merge_name_variants(counts):
    """
    Merges detected names that are word-level variants of each other
    (e.g. 'dracula' and 'count dracula', or 'van helsing' and
    'dr van helsing') into whichever variant was mentioned most often.

    This is a purely structural rule - no hardcoded titles or vocabulary
    - so it generalizes across any book: NER tends to fragment a single
    character across a bare name plus several title/epithet-prefixed
    variants, and this recombines them.
    """
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    canonical = []  # list of (name, normalized_words)
    merged = {}

    for name, count in ordered:
        words = tuple(_normalize_word(word) for word in name.split())

        target = next(
            (
                canonical_name for canonical_name, canonical_words in canonical
                if _is_word_subsequence(words, canonical_words)
                or _is_word_subsequence(canonical_words, words)
            ),
            None,
        )

        if target is not None:
            merged[target] += count
        else:
            canonical.append((name, words))
            merged[name] = count

    return merged


def count_character_names(text, nlp, threshold=1):
    doc = nlp(text)

    names = [clean_name(ent.text) for ent in doc.ents if ent.label_ == 'PERSON']
    names = [name.lower().replace('\n', ' ') for name in names if name]
    counts = _merge_name_variants(Counter(names))

    filtered = {
        name: count for name, count in counts.items()
        if count >= threshold
    }

    return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))


def resolve_title_references(chunks, name_counts, return_title_map=False):
    """
    Attributes bare title mentions (e.g. "the Count") to a character whose
    own name was ever seen directly glued to that title somewhere in the
    text (e.g. "Count Dracula"), when that anchor is unique to one
    character.

    Co-occurrence with other characters mentioned nearby is not a
    reliable signal for this: passages about a title-only character are
    usually narrated by, or mention, the other characters present, not
    the title-bearer's own (rare) name, so nearby-name statistics point
    away from the right answer rather than toward it. A direct textual
    anchor (title and name glued together at least once) is a much
    stronger signal, and it generalizes across books without hardcoding
    anything book specific, since the title list is a small set of
    common English honorifics.

    If `return_title_map` is set, also returns a dict mapping each
    resolved character's name to the list of titles attributed to them
    (e.g. {'dracula': ['count']}), so a caller like
    `get_character_snippets` can pull in title-only passages as well,
    not just mention counts.
    """
    resolved = dict(name_counts)
    detected_names = list(name_counts.keys())
    full_text = ' '.join(chunks)
    title_map = {}

    for title in GENERIC_TITLES:
        anchored_names = [
            name for name in detected_names
            if re.search(
                r'\b' + re.escape(title) + r'\s+' + re.escape(name) + r'\b',
                full_text,
                re.IGNORECASE,
            )
        ]

        if len(anchored_names) != 1:
            continue  # no anchor, or ambiguous (more than one candidate)

        target_name = anchored_names[0]
        bare_title_pattern = re.compile(r'\bthe ' + re.escape(title) + r'\b', re.IGNORECASE)
        title_mentions = len(bare_title_pattern.findall(full_text))

        if title_mentions:
            resolved[target_name] += title_mentions
            title_map.setdefault(target_name, []).append(title)

    resolved = dict(sorted(resolved.items(), key=lambda item: item[1], reverse=True))

    if return_title_map:
        return resolved, title_map
    return resolved


def get_character_snippets(chunks, character_names, name_titles=None):
    """
    Collects each character's matching chunks: their first 2 chronological
    appearances, then the rest ranked by mention density (most mentions
    first).

    Book order alone favors whichever chunk happens to match first, which
    is often front matter: a book's title page contains the protagonist's
    name once, and a table of contents lists every character's name or
    epithet once too, so taking the first N matches tends to hand the LLM
    a title page and a preface instead of narrative content. Ranking by
    how densely a chunk mentions the character routes around that without
    hardcoding anything book specific.

    But density alone has its own failure mode: a character's actual
    introduction usually states their name once and then switches to
    pronouns, so a later, more repetitive scene can outrank it on density
    even though the introduction is disproportionately likely to contain
    the character's core physical/personality description. Worse for a
    first person narrator, who rarely says their own name at all in their
    own narration. Always keeping a character's first 2 chronological
    matches (regardless of density) is a direct, structural fix for that.

    That reopens the front matter problem above for a character whose
    name is unlucky enough to also appear early for a non narrative
    reason (the book's own title, a table of contents entry), so a chunk
    is excluded from being anyone's "guaranteed first appearance" if it
    is also an early match for several other characters at once. A real
    scene is about the handful of people actually in it; a chunk that
    name-drops many different tracked characters together is
    characteristic of front matter, not narrative. This still doesn't
    hardcode anything book specific, since it never has to detect what
    front matter looks like, only that a chunk is a hub shared by many
    characters' earliest matches.

    Matching uses word boundaries (not a plain substring check) so a
    title like "the Count" doesn't also match unrelated text like "the
    country".
    """
    name_titles = name_titles or {}
    patterns = {
        name: [re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)] + [
            re.compile(r'\bthe\s+' + re.escape(title) + r'\b', re.IGNORECASE)
            for title in name_titles.get(name, [])
        ]
        for name in character_names
    }

    scored = {name: [] for name in character_names}
    for chunk in chunks:
        for name, name_patterns in patterns.items():
            mentions = sum(len(pattern.findall(chunk)) for pattern in name_patterns)
            if mentions:
                scored[name].append((mentions, chunk))

    # How many distinct characters mention each chunk at all. A chunk
    # shared by several characters this early is a hub (front matter),
    # not a real introduction for any one of them.
    HUB_MIN_CHARACTERS = 2
    chunk_character_counts = Counter()
    for hits in scored.values():
        for _, chunk in hits:
            chunk_character_counts[chunk] += 1
    hub_chunks = {
        chunk for chunk, count in chunk_character_counts.items()
        if count >= HUB_MIN_CHARACTERS
    }

    result = {}
    for name, hits in scored.items():
        guaranteed = []
        remainder = []

        for mentions, chunk in hits:
            if len(guaranteed) < 2 and chunk not in hub_chunks:
                guaranteed.append((mentions, chunk))
            else:
                remainder.append((mentions, chunk))

        remainder.sort(key=lambda item: item[0], reverse=True)
        result[name] = [chunk for _, chunk in guaranteed + remainder]

    return result
