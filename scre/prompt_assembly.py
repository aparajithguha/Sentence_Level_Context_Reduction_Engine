"""
prompt_assembly.py
==================
Rebuilds the reduced prompt from the kept blocks: original text, original
order, nothing reworded. Trailing spaces and runs of blank lines are trimmed
outside code-like blocks.
"""
from __future__ import annotations

import re

from .blocks import Block
from .config import PromptConfig

_CODE_KINDS = frozenset({"code", "json", "markup", "table"})


def assemble(blocks: list[Block], keep: dict[int, bool], cfg: PromptConfig) -> str:
    parts = []
    for b in blocks:
        if not keep[b.id]:
            continue
        t = b.text
        if cfg.normalize_whitespace and b.kind not in _CODE_KINDS:
            t = re.sub(r"[ \t]+(?=\r?\n)", "", t)
            t = re.sub(r"(?:\r?\n){3,}", "\n\n", t)
        parts.append(t)
    return "".join(parts).rstrip()
