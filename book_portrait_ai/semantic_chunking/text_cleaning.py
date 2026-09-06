import re


def remove_gutenberg_wrapper(text: str) -> str:
    """Removes typical Project Gutenberg header and footer sections."""
    start_pattern = re.compile(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        flags=re.IGNORECASE | re.DOTALL,
    )
    end_pattern = re.compile(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        flags=re.IGNORECASE | re.DOTALL,
    )

    start_match = start_pattern.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = end_pattern.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text


def clean_text(text: str, strip_gutenberg: bool = True) -> str:
    """Normalizes text and removes typical extraction artifacts."""
    text = (
        text.replace("\ufeff", "")
        .replace("\u00ad", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\t", "    ")
    )

    if strip_gutenberg:
        text = remove_gutenberg_wrapper(text)

    # Joins words split by PDF line breaks.
    text = re.sub(
        r"(?<=[A-Za-zÀ-ÿ])-\n(?=[A-Za-zÀ-ÿ])",
        "",
        text,
    )

    # Removes trailing whitespace at the end of lines.
    text = "\n".join(
        re.sub(r"[ \t]+$", "", line)
        for line in text.splitlines()
    )

    # Reduces multiple blank lines to exactly one blank line.
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    return text.strip()
