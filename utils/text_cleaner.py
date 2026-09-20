import re


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line breaks from extracted PDF text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
