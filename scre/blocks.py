"""
blocks.py
=========
Lossless structure reader for prompts.

``parse_blocks`` splits raw prompt text into an ordered list of ``Block``
leaves -- headings, paragraphs, list items, fenced code, JSON, HTML markup,
tables, tag open/close lines, and labels -- whose ``[start, end)`` spans
partition the input exactly: concatenating every block's ``text`` gives back
the original string. Nothing is rewritten; a later stage only decides which
whole blocks to keep.

Structure is recovered from the text itself (markdown headings, XML-style
container tags on their own lines, bullet/number markers, fences, bracket
matching) with no model involved. Each block records its ``parent`` (the
heading or tag container that encloses it), the ``path`` of enclosing titles,
and the current ``label`` (a bold or ``Title:`` line introducing it), which
the role classifier uses.

Code-like blocks (fences, JSON, HTML) are single leaves: they are kept or
dropped whole and never split by sentence.
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass

HTML_TAGS = frozenset(
    "html head body div span p a ul ol li table thead tbody tfoot tr td th style script svg "
    "img br hr button input form label section header footer nav main h1 h2 h3 h4 h5 h6 link "
    "meta title code pre em strong b i canvas iframe template select option textarea video "
    "audio source details summary article aside center font".split()
)
_VOID_TAGS = frozenset("br hr img input link meta source".split())

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_HR = re.compile(r"^\s*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_ITEM = re.compile(r"^([ \t]*)([-*+]|\d+[.)])[ \t]+\S")
_TAG_OPEN = re.compile(r"^\s*<([A-Za-z_][\w:.\-]*)(?:\s[^<>]*)?>\s*$")
_TAG_CLOSE = re.compile(r"^\s*</([A-Za-z_][\w:.\-]*)>\s*$")
_HTML_OPEN = re.compile(r"^\s*(?:<!DOCTYPE\b|<([A-Za-z][\w:\-]*)\b[^>]*>)", re.IGNORECASE)
_BOLD_LABEL = re.compile(r"^\s*\*\*([^*\n]{2,60}?)\*\*:?\s*$")
_COLON_LABEL = re.compile(r"^\s*([A-Z][\w &/'\-]{1,50}):\s*$")

_MAX_JSON_SCAN = 300_000


@dataclass
class Block:
    id: int
    kind: str  # heading|paragraph|item|code|json|markup|table|hr|tag_open|tag_close|label
    start: int = 0
    end: int = 0
    text: str = ""
    parent: int | None = None
    path: tuple = ()          # lowercased titles/tag names of enclosing containers
    label: str = ""           # label line currently introducing this block
    name: str = ""            # heading title / tag name / label text
    level: int = 0            # heading level
    ordered: bool = False     # item: numbered
    depth: int = 0            # item nesting (indent columns)
    list_id: int | None = None  # id of the first item of the contiguous list run
    pair: int | None = None   # tag_open <-> tag_close block ids
    role: str = ""
    role_reason: str = ""


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _find_tag_close(lines: list[str], i: int, name: str) -> int | None:
    open_re = re.compile(rf"^\s*<{re.escape(name)}(?:\s[^<>]*)?>\s*$")
    close_re = re.compile(rf"^\s*</{re.escape(name)}>\s*$")
    depth = 1
    for j in range(i + 1, len(lines)):
        if close_re.match(lines[j]):
            depth -= 1
            if depth == 0:
                return j
        elif open_re.match(lines[j]):
            depth += 1
    return None


def _find_html_end(lines: list[str], i: int, name: str | None) -> int | None:
    if name is None:
        for j in range(i, len(lines)):
            if re.search(r"</html\s*>", lines[j], re.IGNORECASE):
                return j
        return None
    if name in _VOID_TAGS:
        return i
    o = re.compile(rf"<{re.escape(name)}\b", re.IGNORECASE)
    c = re.compile(rf"</{re.escape(name)}\s*>", re.IGNORECASE)
    depth = 0
    for j in range(i, len(lines)):
        depth += len(o.findall(lines[j])) - len(c.findall(lines[j]))
        if depth <= 0:
            return j
    return None


def _scan_json(text: str, start: int) -> int | None:
    depth = 0
    in_str = False
    esc = False
    stop = min(len(text), start + _MAX_JSON_SCAN)
    for k in range(start, stop):
        ch = text[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch == "\n":
                return None
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return k + 1
            if depth < 0:
                return None
    return None


def _is_construct_start(line: str) -> bool:
    return bool(
        _FENCE.match(line) or _HEADING.match(line) or _HR.match(line) or _ITEM.match(line)
        or _TAG_OPEN.match(line) or _TAG_CLOSE.match(line) or line.lstrip().startswith("|")
        or _BOLD_LABEL.match(line) or _COLON_LABEL.match(line)
    )


def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines(keepends=True)
    n = len(lines)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))

    blocks: list[Block] = []
    stack: list[dict] = []  # containers: {'kind': 'heading'|'tag', 'id', 'level', 'name', 'open_block'}
    state = {"label": "", "run_parent": None, "run_first": None}

    def add(kind: str, i0: int, i1: int, **kw) -> Block:
        # A label introduces the block(s) right after it: its own list run,
        # or the single leaf that follows -- not whatever comes after the list.
        if kind != "item" and blocks and blocks[-1].kind == "item":
            state["label"] = ""
        b = Block(id=len(blocks), kind=kind, start=offs[i0], end=offs[i1 + 1])
        b.parent = stack[-1]["id"] if stack else None
        b.path = tuple(e["name"] for e in stack)
        b.label = state["label"]
        for k, v in kw.items():
            setattr(b, k, v)
        blocks.append(b)
        if kind != "item":
            state["run_first"] = None
        return b

    def clear_label():
        state["label"] = ""

    i = 0
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue

        m = _FENCE.match(line)
        if m:
            fence = m.group(1)
            j = i + 1
            while j < n and not (lines[j].strip().startswith(fence[0] * len(fence)) and set(lines[j].strip()) <= {fence[0]}):
                j += 1
            j = min(j, n - 1)
            add("code", i, j)
            clear_label()
            i = j + 1
            continue

        m = _TAG_CLOSE.match(line)
        if m and any(e["kind"] == "tag" and e["name"] == m.group(1).lower() for e in stack):
            name = m.group(1).lower()
            while stack:
                e = stack.pop()
                if e["kind"] == "tag" and e["name"] == name:
                    break
            b = add("tag_close", i, i, name=name)
            for cand in reversed(blocks[:-1]):
                if cand.kind == "tag_open" and cand.name == name and cand.pair is None:
                    cand.pair, b.pair = b.id, cand.id
                    break
            clear_label()
            i += 1
            continue

        m = _HTML_OPEN.match(line)
        if m and (m.group(1) is None or m.group(1).lower() in HTML_TAGS):
            name = m.group(1).lower() if m.group(1) else None
            j = _find_html_end(lines, i, name)
            if j is None:
                j = i
                while j + 1 < n and lines[j + 1].strip():
                    j += 1
            add("markup", i, j)
            clear_label()
            i = j + 1
            continue

        m = _TAG_OPEN.match(line)
        if m and m.group(1).lower() not in HTML_TAGS:
            name = m.group(1).lower()
            if _find_tag_close(lines, i, m.group(1)) is not None:
                b = add("tag_open", i, i, name=name)
                stack.append({"kind": "tag", "id": b.id, "level": 0, "name": name})
                clear_label()
                i += 1
                continue

        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack and stack[-1]["kind"] == "heading" and stack[-1]["level"] >= level:
                stack.pop()
            title = re.sub(r"[`*_]", "", m.group(2)).strip()
            b = add("heading", i, i, level=level, name=title.lower())
            stack.append({"kind": "heading", "id": b.id, "level": level, "name": title.lower()})
            clear_label()
            i += 1
            continue

        if _HR.match(line):
            add("hr", i, i)
            clear_label()
            i += 1
            continue

        if s.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            j = i
            while j + 1 < n and lines[j + 1].strip().startswith("|"):
                j += 1
            add("table", i, j)
            clear_label()
            i = j + 1
            continue

        m = _ITEM.match(line)
        if m:
            ind = _indent(line)
            j = i
            while j + 1 < n and lines[j + 1].strip() and not _is_construct_start(lines[j + 1]):
                j += 1
            ordered = bool(re.match(r"\d", m.group(2)))
            b = add("item", i, j, ordered=ordered, depth=ind)
            pid = stack[-1]["id"] if stack else None
            first = blocks[state["run_first"]] if state["run_first"] is not None else None
            # A new list starts after a non-item block, under a different
            # container, or when a top-level marker switches bullet <-> number.
            if first is None or state["run_parent"] != pid or (first.ordered != ordered and ind <= first.depth):
                state["run_first"], state["run_parent"] = b.id, pid
            b.list_id = state["run_first"]
            i = j + 1
            continue

        if s[0] in "{[":
            start = offs[i] + (len(line) - len(line.lstrip()))
            end = _scan_json(text, start)
            if end is not None:
                sub = text[start:end]
                ok = False
                try:
                    json.loads(sub)
                    ok = True
                except ValueError:
                    ok = sub.count("\n") >= 2 and re.search(r'"\s*:', sub) is not None
                if ok:
                    j = bisect.bisect_right(offs, end - 1) - 1
                    add("json", i, min(j, n - 1))
                    clear_label()
                    i = min(j, n - 1) + 1
                    continue

        m = _BOLD_LABEL.match(line) or _COLON_LABEL.match(line)
        if m:
            b = add("label", i, i, name=m.group(1).strip().lower().rstrip(":").strip())
            state["label"] = b.name
            i += 1
            continue

        j = i
        while j + 1 < n and lines[j + 1].strip() and not _is_construct_start(lines[j + 1]):
            j += 1
        add("paragraph", i, j)
        clear_label()
        i = j + 1

    if not blocks:
        # Whitespace-only input still partitions exactly: one paragraph block.
        return [Block(id=0, kind="paragraph", start=0, end=len(text), text=text)] if text else []
    blocks[0].start = 0
    for k in range(len(blocks) - 1):
        blocks[k].end = blocks[k + 1].start
    blocks[-1].end = len(text)
    for b in blocks:
        b.text = text[b.start:b.end]
    return blocks


def rebuild(blocks: list[Block], keep: set[int] | None = None) -> str:
    """Concatenate block texts (all of them, or only ids in ``keep``)."""
    return "".join(b.text for b in blocks if keep is None or b.id in keep)
