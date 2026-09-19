"""
prompt_policy.py
================
Decides which blocks of a prompt to keep. The default is to keep everything;
only these are ever dropped:

1. horizontal rules;
2. an instruction that near-duplicates an earlier one;
3. examples beyond the first ``PromptConfig.example_keep`` of each sibling group;
4. whole modules a given task cannot use (only when ``drop_blocks`` is passed);
5. headings, labels and tag pairs left empty by the drops above.

Instructions, identity, formats, reasoning steps, numbered steps, tool
definitions and every code/JSON/HTML block are never dropped.
"""
from __future__ import annotations

from collections import defaultdict

from .blocks import Block
from .config import PromptConfig
from .prompt_graph import PromptGraph


def decide(
    blocks: list[Block], graph: PromptGraph, cfg: PromptConfig, drop_blocks: dict[int, str] | None = None,
) -> tuple[dict[int, bool], dict[int, str]]:
    """``drop_blocks`` (block id -> reason) are whole task-unneeded modules,
    already vetted by ``prompt_modules``; they are dropped before orphan cleanup."""
    keep = {b.id: True for b in blocks}
    reason: dict[int, str] = {}

    def drop(i: int, why: str) -> None:
        if keep[i]:
            keep[i] = False
            reason[i] = why

    if cfg.drop_horizontal_rules:
        for b in blocks:
            if b.role == "filler":
                drop(b.id, "horizontal rule")

    for later, earlier in graph.duplicate_of.items():
        drop(later, f"duplicate of block {earlier}")

    groups: dict[tuple, list] = defaultdict(list)
    for u in graph.example_units:
        groups[u.key].append(u)
    for us in groups.values():
        for u in us[cfg.example_keep:]:
            for i in u.blocks:
                drop(i, "example trimmed")

    for i, why in (drop_blocks or {}).items():
        drop(i, why)

    # Subtree extent of every container, for orphan cleanup.
    last_descendant: dict[int, int] = {}
    for b in blocks:
        p = b.parent
        while p is not None:
            last_descendant[p] = max(last_descendant.get(p, p), b.id)
            p = blocks[p].parent

    changed = True
    while changed:
        changed = False
        for b in blocks:
            if not keep[b.id]:
                continue
            if b.kind == "tag_open" and b.pair is not None:
                inner = range(b.id + 1, b.pair)
                if len(inner) and not any(keep[i] for i in inner):
                    drop(b.id, "empty container")
                    drop(b.pair, "empty container")
                    changed = True
            elif b.kind == "label":
                scope = []
                k = b.id + 1
                while k < len(blocks) and blocks[k].label == b.name:
                    scope.append(k)
                    k += 1
                if scope and not any(keep[i] for i in scope):
                    drop(b.id, "empty label")
                    changed = True
            elif b.kind == "heading" and b.id in last_descendant:
                inner = range(b.id + 1, last_descendant[b.id] + 1)
                if not any(keep[i] and blocks[i].kind != "heading" for i in inner):
                    drop(b.id, "empty heading")
                    changed = True
    return keep, reason
