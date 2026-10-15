from __future__ import annotations

import re

# Keeps numbers like 60,922 / 26.97 intact so exact-number queries match in BM25.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def clean(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u2019", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
