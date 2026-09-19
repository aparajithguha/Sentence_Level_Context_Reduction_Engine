"""
roles.py
========
Assigns each ``Block`` a role -- what it *is* in a prompt -- from the block's
own text, its enclosing headings/tags, and the label that introduces it. Pure
pattern matching; no model.

Roles: ``heading``, ``tag`` (container open/close line), ``label``,
``filler`` (horizontal rule), ``code`` (fenced/HTML/table), ``tool_def``
(JSON tool schema or text inside a tools/functions section), ``example``,
``identity``, ``procedure`` (numbered steps or a workflow/steps section),
``format``, ``reasoning``, ``reference``, ``rules``, and ``text`` (anything
else). Only ``example`` and ``filler`` can ever be dropped by the policy, so a
wrong label on any other role costs nothing but a less specific report.
"""
from __future__ import annotations

import re

from .blocks import Block

RULE_RE = re.compile(
    r"\b(must|never|always|do not|don't|dont|cannot|can't|shall|should|required|requires?|"
    r"forbidden|prohibited|not allowed|avoid|ensure|make sure|only|unless|except|refuse|strictly)\b",
    re.IGNORECASE,
)
STRONG_RULE_RE = re.compile(
    r"\b(must|must not|never|always|do not|don't|cannot|can't|shall not|should not|shouldn't|forbidden|prohibited|not allowed)\b",
    re.IGNORECASE,
)

_EX_TAG = re.compile(r"^(?:good_|bad_)?examples?(?:_\w+)?$|^\w+_examples?$|^samples?$|^few_?shot$|^demo(?:nstration)?s?$", re.IGNORECASE)
_EX_TITLE = re.compile(r"^(?:good |bad |some |more |few[- ]shot |sample |example )*(?:examples?|samples?|demonstrations?|example interactions?)\b", re.IGNORECASE)
_IDENTITY = re.compile(r"^\s*(?:[-*]\s*)?(?:you are|you're|your role|as an? (?:ai|assistant)|act as)\b", re.IGNORECASE)
_TOOL_TITLE = re.compile(r"\b(tools?|functions?|apis?|commands?|namespace)\b", re.IGNORECASE)
_FORMAT_TITLE = re.compile(r"\b(output|response|answer|reply)\s+format|\bformat(?:ting)?\b|\bstyle\b|\btone\b", re.IGNORECASE)
_REASONING_TITLE = re.compile(r"\b(reasoning|thinking|thoughts?|chain[- ]of[- ]thought|scratchpad|reflection)\b", re.IGNORECASE)
_PROCEDURE_TITLE = re.compile(r"\b(workflow|steps?|procedure|process|how to|your task|approach)\b", re.IGNORECASE)
_REFERENCE_TITLE = re.compile(r"\b(user (?:profile|bio|info)|memory|knowledge cutoff|current date|profile)\b", re.IGNORECASE)
_TOOL_JSON = re.compile(r'"(?:parameters|input_schema|arguments)"')


def example_scope(b: Block, blocks: list[Block]) -> str:
    """Return 'tag' | 'heading' | 'label' when ``b`` sits inside an example, else ''."""
    if b.kind in ("tag_open", "tag_close") and _EX_TAG.match(b.name):
        return "tag"
    p = b.parent
    while p is not None:
        c = blocks[p]
        if c.kind == "tag_open" and _EX_TAG.match(c.name):
            return "tag"
        if c.kind == "heading" and _EX_TITLE.match(c.name):
            return "heading"
        p = c.parent
    if b.label and _EX_TITLE.match(b.label):
        return "label"
    if b.kind == "label" and _EX_TITLE.match(b.name):
        return "label"
    return ""


def _context_titles(b: Block, blocks: list[Block]) -> list[str]:
    titles = [b.label] if b.label else []
    p = b.parent
    while p is not None:
        titles.append(blocks[p].name)
        p = blocks[p].parent
    return titles


def classify(blocks: list[Block]) -> None:
    for b in blocks:
        b.role, b.role_reason = _role(b, blocks)


def _role(b: Block, blocks: list[Block]) -> tuple[str, str]:
    if b.kind == "heading":
        return "heading", ""
    if b.kind in ("tag_open", "tag_close"):
        return "tag", b.name
    if b.kind == "hr":
        return "filler", "horizontal rule"
    scope = example_scope(b, blocks)
    if b.kind == "label":
        return ("example", f"example label ({scope})") if scope else ("label", b.name)
    if scope:
        return "example", f"inside example ({scope})"
    titles = _context_titles(b, blocks)
    if b.kind in ("json", "code", "markup", "table"):
        if b.kind == "json" and (_TOOL_JSON.search(b.text) or any(_TOOL_TITLE.search(t) for t in titles)):
            return "tool_def", "json tool schema"
        return "code", b.kind
    text = b.text
    if _IDENTITY.match(text):
        return "identity", "starts with 'You are'"
    if b.kind == "item" and b.ordered:
        return "procedure", "numbered step"
    for t in titles:
        if _TOOL_TITLE.search(t):
            return "tool_def", f"under '{t}'"
        if _FORMAT_TITLE.search(t):
            return "format", f"under '{t}'"
        if _REASONING_TITLE.search(t):
            return "reasoning", f"under '{t}'"
        if _PROCEDURE_TITLE.search(t):
            return "procedure", f"under '{t}'"
        if _REFERENCE_TITLE.search(t):
            return "reference", f"under '{t}'"
    if RULE_RE.search(text):
        return "rules", "rule wording"
    return "text", ""
