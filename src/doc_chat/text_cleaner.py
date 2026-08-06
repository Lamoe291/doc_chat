import re

def clean_text(text: str) -> str:
    """
    Basic cleaning for extracted PDF text.
    """

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text