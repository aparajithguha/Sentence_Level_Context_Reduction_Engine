"""
prompt_modules.py
=================
Splits a parsed prompt into task-selectable *modules* and decides which ones
a given task can do without. Deterministic and model-free; the caller (or an
LLM behind the caller's ``drop_modules``) may propose drops instead, but the
guards here always apply.

A module is a heading section or a tag container. A container that is large
and has two or more child containers is split into those children (its own
lead-in text stays); anything else is one module. Modules are dropped whole
and never rewritten.

Guards -- a module is kept when any of these holds, whoever proposed the drop:
  * its title names a global concern (role, safety, format, rules, ...);
  * it holds an identity ("You are ...") block;
  * it is small (``PromptConfig.module_min_chars``);
  * the task names it; for the built-in matcher, also when the task uses a term
    found in only some of the modules;
  * the remaining text, or the task, uses an identifier the section defines;
  * the section defines vocabulary that the prompt's own examples use;
  * it is the last section left that shows how tools are defined or called.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .blocks import Block
from .config import PromptConfig

_CONTAINER = ("heading", "tag_open")
_GLOBAL_NAME = re.compile(
    r"role|identity|persona|safety|security|privacy|polic|guardrail|refus|harm|copyright|style|tone|"
    r"format|output|response|communicat|general|core|principle|important|overview|introduction|about|"
    r"instruction|guideline|rules?\b|constraint|behavio|critical",
    re.IGNORECASE,
)
_TOOL_NAME = re.compile(r"tool|function|api|namespace|browser|python|bash|shell|git|search|mcp|plugin|command", re.IGNORECASE)
_STOP = frozenset(
    "the and for with that this from your you are was were has have had not but can will would should could "
    "into onto over out all any some one two its it's our their them then than also just only very more most "
    "please using use used via per".split()
)


_SYMBOL = re.compile(r"`([A-Za-z_][\w.\-]{2,})`|\b([a-z]+(?:_[a-z0-9]+)+)\b|\b([a-z]+[A-Z][A-Za-z0-9]*)\b|</?([A-Za-z][\w\-]{2,})>")


# Typed tool definitions and call syntax: they teach the agent HOW any tool is called, by example.
_SIGNATURE = re.compile(
    r"\bnamespace\s+\w+\s*\{|=>\s*any\b|\btype\s+\w+\s*=\s*\(|\"parameters\"\s*:|\binput_schema\b|"
    r"\[tool_call:|<tool_call>|\bto=[\w.]+"
)


def symbol_counts(text: str) -> Counter:
    """Identifier-like terms: `backticked`, snake_case, camelCase, <tags>. Lowercased."""
    return Counter(next(g for g in m.groups() if g).lower() for m in _SYMBOL.finditer(text))


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9]{2,}", re.sub(r"[_\-./]", " ", text.lower()))
    out = set()
    for w in words:
        if w in _STOP:
            continue
        for suf in ("ing", "es", "ed", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


@dataclass
class Module:
    id: int                      # id of the heading / tag_open block
    name: str
    path: tuple
    block_ids: list[int]         # container line, everything inside it, and its closing tag
    chars: int
    cls: str                     # "global" | "tool" | "other"
    preview: str
    tokens: frozenset = field(default_factory=frozenset)
    name_tokens: frozenset = field(default_factory=frozenset)
    defines: frozenset = field(default_factory=frozenset)   # identifiers this section introduces or leans on
    is_example: bool = False
    body: str = ""               # the section's text, blocks joined by newlines
    has_signature: bool = False  # shows a typed tool definition or a tool-call form


def find_modules(blocks: list[Block], cfg: PromptConfig) -> list[Module]:
    kids: dict[int | None, list[int]] = defaultdict(list)
    members: dict[int, list[int]] = {}
    for b in blocks:
        if b.kind in _CONTAINER:
            kids[b.parent].append(b.id)
            members[b.id] = [b.id]
    for b in blocks:
        p = b.parent
        while p is not None:
            members[p].append(b.id)
            p = blocks[p].parent
        if b.kind == "tag_open" and b.pair is not None:
            members[b.id].append(b.pair)
    chars = {c: sum(len(blocks[i].text) for i in set(ids)) for c, ids in members.items()}

    out: list[Module] = []

    def make(cid: int) -> Module:
        ids = sorted(set(members[cid]))
        body = [blocks[i] for i in ids if blocks[i].kind not in ("heading", "tag_open", "tag_close", "label", "hr")]
        text = " ".join(b.text for b in body)
        name = blocks[cid].name
        tool_chars = sum(len(b.text) for b in body if b.role == "tool_def")
        if _GLOBAL_NAME.search(name) or any(b.role == "identity" for b in body):
            cls = "global"
        elif _TOOL_NAME.search(name) or tool_chars > 0.5 * max(len(text), 1):
            cls = "tool"
        else:
            cls = "other"
        preview = re.sub(r"\s+", " ", body[0].text).strip()[:240] if body else ""
        counts = symbol_counts(text)
        lead = set(symbol_counts(name + " " + (body[0].text if body else "")))
        squash = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())   # headings lose underscores: `apply_patch` -> "applypatch"
        title = squash(name)
        defines = frozenset(sym for sym, c in counts.items() if c >= 2 or sym in lead or (title and squash(sym) == title))
        return Module(cid, name, blocks[cid].path, ids, chars[cid], cls, preview,
                      frozenset(tokens(text)), frozenset(tokens(name)), defines,
                      bool(body) and all(b.role == "example" for b in body),
                      "\n".join(b.text.strip() for b in body), bool(_SIGNATURE.search(text)))

    def walk(cids: list[int]) -> None:
        for cid in cids:
            sub = kids.get(cid, [])
            if len(sub) >= 2 and chars[cid] > cfg.module_split_chars:
                walk(sub)
            else:
                out.append(make(cid))

    walk(kids.get(None, []))
    return out


def _mentioned(m: Module, terms: set[str], df: Counter, max_df: int, n_modules: int, by_content: bool = True) -> str:
    named = sorted(m.name_tokens & terms)
    if named:
        return f"task names this section ({', '.join(named[:3])})"
    if not by_content:
        return ""
    hits = sorted(t for t in terms if t in m.tokens and df[t] <= max_df)
    if hits:
        return f"task term '{hits[0]}' occurs in only {df[hits[0]]} of {n_modules} sections"
    return ""


def _apply_reference_guards(
    modules: list[Module], result: dict[int, tuple[bool, str]], task: str, outside_text: str, output_vocab: frozenset,
) -> None:
    """Flip drops back to keeps, in place, until stable:
      * a section that defines an identifier the kept text (or the task) uses;
      * a section that defines vocabulary the prompt's own examples use (the agent's answers need it).
    Both are cross-references in the prompt's own text, not guesses about the task."""
    used = set(symbol_counts(outside_text)) | set(symbol_counts(task))
    for m in modules:
        if result[m.id][0]:
            used |= m.defines
    changed = True
    while changed:
        changed = False
        for m in modules:
            if result[m.id][0]:
                continue
            hit = sorted(m.defines & used)
            vocab = sorted(m.defines & output_vocab) if not m.is_example else []
            if hit:
                result[m.id] = (True, f"kept text uses `{hit[0]}`, which this section defines")
            elif vocab:
                result[m.id] = (True, f"defines `{vocab[0]}`, which the prompt's own examples use")
            else:
                continue
            used |= m.defines
            changed = True


def _apply_convention_guard(modules: list[Module], result: dict[int, tuple[bool, str]]) -> None:
    """If every section that shows how tools are defined or called was dropped, keep the smallest one:
    the remaining tool sections would otherwise lose the calling convention they all share
    (Round 4: the agent stopped making tool calls once all sibling definitions were gone)."""
    sig = [m for m in modules if m.has_signature]
    if sig and not any(result[m.id][0] for m in sig):
        keep = min(sig, key=lambda m: m.chars)
        result[keep.id] = (True, "kept as the one example of how tools are defined and called")


def select_modules(
    modules: list[Module], task: str | None, cfg: PromptConfig, proposed_drop: set[int] | None = None,
    outside_text: str = "", output_vocab: frozenset = frozenset(),
) -> dict[int, tuple[bool, str]]:
    """Decide keep/drop per module id. ``proposed_drop`` (from a caller) replaces
    the built-in task matcher, but the guards still force-keep modules."""
    terms = tokens(task or "")
    df: Counter = Counter()
    for m in modules:
        df.update(m.tokens)
    max_df = max(1, int(len(modules) * cfg.module_max_df_ratio))
    result: dict[int, tuple[bool, str]] = {}
    for m in modules:
        if m.cls == "global":
            result[m.id] = (True, "global section")
        elif m.chars < cfg.module_min_chars:
            result[m.id] = (True, "small section")
        elif proposed_drop is None and task is None:
            result[m.id] = (True, "no task given")
        else:
            # A caller's proposal is only overridden when the task names the section;
            # the built-in matcher also counts task terms that occur in the section.
            why = _mentioned(m, terms, df, max_df, len(modules), by_content=proposed_drop is None) if terms else ""
            if why:
                result[m.id] = (True, why)
            elif proposed_drop is not None and m.id not in proposed_drop:
                result[m.id] = (True, "selector kept it")
            else:
                result[m.id] = (False, "selector dropped it" if proposed_drop is not None else "no task term matches this section")
    # Guards feed each other (a restored section can define identifiers the kept text now uses), so repeat to a fixpoint.
    while True:
        before = {i: r[0] for i, r in result.items()}
        _apply_reference_guards(modules, result, task or "", outside_text, output_vocab)
        _apply_convention_guard(modules, result)
        if before == {i: r[0] for i, r in result.items()}:
            break
    return result
