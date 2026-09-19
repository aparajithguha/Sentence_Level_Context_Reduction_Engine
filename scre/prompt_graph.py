"""
prompt_graph.py
===============
The relationships between a prompt's blocks that the keep-policy relies on:

* ``children`` -- containment (which blocks sit under which heading/tag).
* ``example_units`` -- example groups: each unit is the set of blocks that
  must be kept or dropped together (a ``<example>...</example>`` container,
  an "Example N:" label and what it introduces, or one item of an
  "Examples:" list), keyed by the group of sibling examples it belongs to.
* ``duplicate_of`` -- a later instruction that repeats an earlier one.

This is the prompt-mode counterpart of ``scre.graph``: that module links
sentences by proximity and shared entities, which suits decision documents;
prompts need structural links instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .blocks import Block
from .config import PromptConfig
from .roles import _EX_TAG, _EX_TITLE, STRONG_RULE_RE

_WORD = re.compile(r"\w+")
_SINGLE_EXAMPLE_TITLE = re.compile(r"^(?:example|sample)\s*[\d#:.\-]|^(?:example|sample)\s+\w+", re.IGNORECASE)
_DUP_ROLES = frozenset({"rules", "text", "identity", "format", "reasoning", "reference"})


@dataclass
class ExampleUnit:
    key: tuple
    blocks: list[int]
    strong_rules: bool


@dataclass
class PromptGraph:
    children: dict
    example_units: list
    duplicate_of: dict


def _subtree_ids(blocks: list[Block], root: int) -> list[int]:
    out = []
    for k in range(root + 1, len(blocks)):
        p = blocks[k].parent
        while p is not None and p != root:
            p = blocks[p].parent
        if p != root:
            break
        out.append(k)
    return out


def _strong(blocks: list[Block], ids: list[int]) -> bool:
    return any(
        blocks[i].kind not in ("tag_open", "tag_close", "heading", "label", "hr") and STRONG_RULE_RE.search(blocks[i].text)
        for i in ids
    )


def _item_units(blocks: list[Block], ids: list[int]) -> list[list[int]]:
    """Split a run of blocks into units: each top-level item takes its nested items."""
    items = [i for i in ids if blocks[i].kind == "item"]
    top = min((blocks[i].depth for i in items), default=0)
    units: list[list[int]] = []
    for i in ids:
        b = blocks[i]
        if b.kind == "item" and b.depth > top and units:
            units[-1].append(i)
        else:
            units.append([i])
    return units


def build_graph(blocks: list[Block], cfg: PromptConfig) -> PromptGraph:
    children: dict = {}
    for b in blocks:
        children.setdefault(b.parent if b.parent is not None else -1, []).append(b.id)

    units: list[ExampleUnit] = []
    claimed: set[int] = set()

    def add_unit(key: tuple, ids: list[int]) -> None:
        ids = [i for i in ids if i not in claimed]
        if ids:
            claimed.update(ids)
            units.append(ExampleUnit(key, ids, _strong(blocks, ids)))

    # (a) <example> ... </example> containers. A container that itself holds
    # example containers (e.g. <examples> around several <example>) is only
    # a wrapper: its children are the units, grouped under the wrapper.
    for b in blocks:
        if b.kind == "tag_open" and b.pair is not None and _EX_TAG.match(b.name) and b.id not in claimed:
            inner = range(b.id + 1, b.pair)
            if any(blocks[i].kind == "tag_open" and blocks[i].pair is not None and _EX_TAG.match(blocks[i].name) for i in inner):
                continue
            add_unit((b.parent, "tag", b.name), list(range(b.id, b.pair + 1)))

    # (b) "Example(s):" / "Example N:" labels and what they introduce
    for b in blocks:
        if b.kind == "label" and _EX_TITLE.match(b.name) and b.id not in claimed:
            scope = []
            k = b.id + 1
            while k < len(blocks) and blocks[k].label == b.name and k not in claimed:
                scope.append(k)
                k += 1
            if not scope:
                continue
            if all(blocks[i].kind == "item" for i in scope):
                parts = _item_units(blocks, scope)
                if len(parts) >= 2:
                    for ids in parts:
                        add_unit((b.parent, "label-list", b.id), ids)
                    claimed.add(b.id)  # the label line itself stays; orphan cleanup handles it
                    continue
            add_unit((b.parent, "label-block"), [b.id] + scope)

    # (c) headings: "Examples" sections, and sibling "Example N" sections
    for b in blocks:
        if b.kind != "heading" or b.id in claimed:
            continue
        if _SINGLE_EXAMPLE_TITLE.match(b.name):
            add_unit((b.parent, "heading-sibling"), [b.id] + _subtree_ids(blocks, b.id))
        elif _EX_TITLE.match(b.name):
            direct = [i for i in children.get(b.id, []) if i not in claimed and blocks[i].kind in ("paragraph", "item", "code", "json")]
            parts = _item_units(blocks, direct)
            if len(parts) >= 2:
                for ids in parts:
                    add_unit((b.id, "heading"), ids)

    # Duplicates: a later instruction repeating an earlier one.
    duplicate_of: dict[int, int] = {}
    seen_exact: dict[str, int] = {}
    seen: list[tuple[int, frozenset]] = []
    for b in blocks:
        if b.kind not in ("item", "paragraph") or b.role not in _DUP_ROLES or b.ordered or b.id in claimed:
            continue
        norm = " ".join(_WORD.findall(b.text.lower()))
        if len(norm) < cfg.duplicate_min_chars:
            continue
        if norm in seen_exact:
            duplicate_of[b.id] = seen_exact[norm]
            continue
        toks = frozenset(norm.split())
        for other_id, other in seen:
            small, big = sorted((len(toks), len(other)))
            if small / big < cfg.duplicate_threshold:
                continue
            if len(toks & other) / len(toks | other) >= cfg.duplicate_threshold:
                duplicate_of[b.id] = other_id
                break
        else:
            seen.append((b.id, toks))
            seen_exact[norm] = b.id
    return PromptGraph(children=children, example_units=units, duplicate_of=duplicate_of)
