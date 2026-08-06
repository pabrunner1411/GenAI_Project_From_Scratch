from functools import lru_cache
import re
from typing import Any

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load an embedding model once and cache it for later calls."""
    return SentenceTransformer(model_name)


def count_tokens(text: str, encoding: Any) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def hard_split_by_tokens(
    text: str,
    encoding: Any,
    max_tokens: int,
) -> list[str]:
    """Fallback for individual units that exceed the maximum size."""
    tokens = encoding.encode(text, disallowed_special=())
    parts: list[str] = []

    for start in range(0, len(tokens), max_tokens):
        part = encoding.decode(tokens[start:start + max_tokens]).strip()
        if part:
            parts.append(part)

    return parts


def split_atomic_units(
    text: str,
    encoding: Any,
    max_unit_tokens: int = 160,
) -> list[str]:
    """
    Create small base units.

    Short paragraphs remain intact. Long paragraphs are split into sentence
    groups. Extremely long sentences are split by token count.
    """
    paragraphs = [
        re.sub(r"[ \t\n]+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    units: list[str] = []

    for paragraph in paragraphs:
        if count_tokens(paragraph, encoding) <= max_unit_tokens:
            units.append(paragraph)
            continue

        sentences = re.split(
            r'(?<=[.!?])\s+(?=(?:["“‘(\[])?[A-ZÄÖÜ])',
            paragraph,
        )

        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = count_tokens(sentence, encoding)

            if sentence_tokens > max_unit_tokens:
                if current_sentences:
                    units.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                units.extend(
                    hard_split_by_tokens(
                        sentence,
                        encoding,
                        max_unit_tokens,
                    )
                )
                continue

            if (
                current_sentences
                and current_tokens + sentence_tokens > max_unit_tokens
            ):
                units.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            units.append(" ".join(current_sentences))

    return [unit for unit in units if unit.strip()]


def semantic_chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 900,
    breakpoint_percentile: float = 20.0,
    overlap_units: int = 1,
    max_unit_tokens: int = 160,
    encoding_name: str = "cl100k_base",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[dict[str, Any]], int, float, int]:
    """
    Split text at semantic topic changes.

    Low similarity between two adjacent units is treated as a possible chunk
    boundary. ``min_tokens`` and ``max_tokens`` constrain the chunk size.
    """
    if min_tokens <= 0:
        raise ValueError("The minimum chunk size must be greater than 0.")

    if max_tokens <= min_tokens:
        raise ValueError(
            "The maximum chunk size must be greater than the minimum."
        )

    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError(
            "The breakpoint percentile must be between 0 and 100."
        )

    if overlap_units < 0:
        raise ValueError("The overlap cannot be negative.")

    encoding = tiktoken.get_encoding(encoding_name)
    units = split_atomic_units(
        text=text,
        encoding=encoding,
        max_unit_tokens=max_unit_tokens,
    )

    if not units:
        return [], 0, 0.0, 0

    unit_token_counts = [
        count_tokens(unit, encoding)
        for unit in units
    ]

    model = get_embedding_model(model_name)
    embeddings = model.encode(
        units,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if len(units) > 1:
        similarities = np.sum(
            embeddings[:-1] * embeddings[1:],
            axis=1,
        )
        semantic_threshold = float(
            np.percentile(similarities, breakpoint_percentile)
        )
    else:
        similarities = np.array([], dtype=np.float32)
        semantic_threshold = -1.0

    boundaries: list[tuple[int, int, str, float | None]] = []

    chunk_start = 0
    current_tokens = unit_token_counts[0]

    for unit_index in range(1, len(units)):
        boundary_similarity = float(similarities[unit_index - 1])

        exceeds_maximum = (
            current_tokens + unit_token_counts[unit_index] > max_tokens
        )
        semantic_shift = (
            current_tokens >= min_tokens
            and boundary_similarity <= semantic_threshold
        )

        if exceeds_maximum or semantic_shift:
            reason = "max_tokens" if exceeds_maximum else "semantic_shift"
            boundaries.append(
                (
                    chunk_start,
                    unit_index,
                    reason,
                    boundary_similarity,
                )
            )
            chunk_start = unit_index
            current_tokens = unit_token_counts[unit_index]
        else:
            current_tokens += unit_token_counts[unit_index]

    boundaries.append(
        (
            chunk_start,
            len(units),
            "document_end",
            None,
        )
    )

    token_prefix = np.concatenate(
        [
            np.array([0]),
            np.cumsum(unit_token_counts),
        ]
    )

    def approximate_token_count(start: int, end: int) -> int:
        return int(token_prefix[end] - token_prefix[start])

    chunks: list[dict[str, Any]] = []

    for chunk_index, boundary in enumerate(boundaries):
        semantic_start, end, break_reason, boundary_similarity = boundary

        expanded_start = semantic_start
        copied_units = 0

        while copied_units < overlap_units and expanded_start > 0:
            candidate_start = expanded_start - 1

            if approximate_token_count(candidate_start, end) > max_tokens:
                break

            expanded_start = candidate_start
            copied_units += 1

        chunk_content = "\n\n".join(
            units[expanded_start:end]
        ).strip()

        internal_similarities = similarities[
            expanded_start:max(expanded_start, end - 1)
        ]

        mean_similarity = (
            float(np.mean(internal_similarities))
            if len(internal_similarities)
            else None
        )

        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_unit": expanded_start,
                "semantic_start_unit": semantic_start,
                "end_unit": end,
                "overlap_units": copied_units,
                "token_count": count_tokens(chunk_content, encoding),
                "character_count": len(chunk_content),
                "mean_internal_similarity": mean_similarity,
                "boundary_similarity": boundary_similarity,
                "break_reason": break_reason,
                "semantic_threshold": semantic_threshold,
                "text": chunk_content,
            }
        )

    total_tokens = count_tokens(text, encoding)

    return chunks, total_tokens, semantic_threshold, len(units)
