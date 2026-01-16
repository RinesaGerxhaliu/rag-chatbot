import re

def clean_text(text: str) -> str:
    """
    Clean raw document text:
    - Remove null characters
    - Normalize line breaks
    - Normalize spaces
    - Remove references/bibliography sections
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lower_text = text.lower()
    for keyword in ["references", "bibliography"]:
        idx = lower_text.rfind(keyword)
        if idx != -1 and idx > len(text) * 0.7:
            text = text[:idx]

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
