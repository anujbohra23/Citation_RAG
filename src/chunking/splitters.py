import re
from typing import List


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def sliding_word_windows(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    windows = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        window = words[start:end]
        if not window:
            continue
        windows.append(" ".join(window))
        if end >= len(words):
            break
    return windows
