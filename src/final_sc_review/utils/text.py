"""Text utilities."""

from __future__ import annotations

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using canonical punctuation rules.

    Matches the groundtruth generator logic from the source repo.
    """
    if text is None:
        return []

    sentences = re.split(r"([.!?]+\s+)", text)
    result: List[str] = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if sentence:
            if i + 1 < len(sentences) and re.match(r"^[.!?]+\s*$", sentences[i + 1]):
                sentence = sentence + sentences[i + 1].strip()
                i += 2
            else:
                i += 1
            result.append(sentence)
        else:
            i += 1
    return result
