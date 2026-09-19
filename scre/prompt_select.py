"""
prompt_select.py
================
LLM-driven section selection for ``PromptReducer.reduce(..., selector=...)`` (approach "B2").

The engine never calls a model. ``llm_selector(ask)`` wraps *your* function -- anything that takes a
chat message list and returns the reply text -- into a selector the reducer can call:

    ask = lambda messages: my_client.chat(messages).text
    result = PromptReducer().reduce(prompt, task="Draw me a bicycle", selector=llm_selector(ask))

How it works: the model sees the task and each section's full text, first lists the steps the agent
must take and the sections each step needs, then names sections that can go. A section used by any
step is never dropped, ids that are not in the outline are ignored, and an unusable reply drops
nothing. The reducer then applies its own deterministic guards to whatever the selector returns,
and rebuilds the prompt verbatim from the sections that remain.

Nothing here imports an LLM SDK.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

Ask = Callable[[list[dict]], str]
Selector = Callable[[list[dict], str], Iterable[int]]

SYSTEM_PROMPT = (
    "You decide which sections of an AI agent's system prompt can be removed for ONE task. The agent knows "
    "nothing except what stays in the prompt: it can only use tools, commands and output formats that the "
    "remaining prompt describes. Work in two steps and reply with JSON only.\n"
    "1. steps: the concrete steps the agent must take to finish the task, including how it delivers its answer "
    "and which tool or format each step needs. For each step give the ids of the sections that describe that "
    "tool, format or rule.\n"
    "2. drop: ids of sections that no step needs and that are not general behavior, safety or style rules "
    "applying to every reply. When unsure, keep the section.\n"
    '{"steps": [{"step": "...", "sections": [ids]}], "drop": [ids]}'
)
_JSON = re.compile(r"\{.*\}", re.S)


def build_messages(outline: list[dict], task: str) -> list[dict]:
    """The chat messages that ask a model which sections the task can do without.
    ``outline`` rows need ``id``, ``name``, ``chars`` and ``text`` (see ``PromptReducer.outline``)."""
    body = "\n\n".join(f"[{m['id']}] {m['name']} ({m['chars']} chars)\n{m.get('text', m.get('preview', ''))}" for m in outline)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK: {task}\n\nSECTIONS:\n{body}"}]


@dataclass
class Selection:
    drop: list[int] = field(default_factory=list)      # ids to drop (after removing step-needed and unknown ids)
    steps: list[str] = field(default_factory=list)     # the model's plan for the task
    needed: list[int] = field(default_factory=list)    # ids some step uses
    proposed: list[int] = field(default_factory=list)  # ids the model listed under "drop", before filtering
    parsed: bool = False                                # False: the reply was not usable JSON -> nothing dropped


def parse_reply(reply: str, outline: list[dict]) -> Selection:
    m = _JSON.search(reply or "")
    if not m:
        return Selection()
    try:
        data = json.loads(m.group(0))
        proposed = [int(i) for i in data.get("drop", [])]
        steps = [s for s in data.get("steps", []) if isinstance(s, dict)]
    except (ValueError, TypeError, AttributeError):
        return Selection()
    needed = {int(i) for s in steps for i in s.get("sections", []) if str(i).lstrip("-").isdigit()}
    valid = {row["id"] for row in outline}
    return Selection(drop=[i for i in proposed if i in valid and i not in needed],
                     steps=[str(s.get("step", "")) for s in steps][:6], needed=sorted(needed),
                     proposed=proposed, parsed=True)


class LLMSelector:
    """Callable ``(outline, task) -> ids to drop`` backed by the caller's ``ask`` function.
    ``last`` keeps the latest selection for inspection (steps, what was dropped and why not)."""

    def __init__(self, ask: Ask, max_section_chars: int = 3000):
        self.ask = ask
        self.max_text_chars = max_section_chars          # the reducer reads this to size each section's text
        self.last: dict[str, Any] = {}

    def __call__(self, outline: list[dict], task: str) -> list[int]:
        sel = parse_reply(self.ask(build_messages(outline, task)), outline)
        names = {row["id"]: row["name"] for row in outline}
        self.last = {"parsed": sel.parsed, "steps": sel.steps,
                     "proposed_drop": [names.get(i, i) for i in sel.proposed],
                     "kept_because_a_step_needs_it": sorted(names.get(i, i) for i in sel.proposed if i in set(sel.needed))}
        return sel.drop


def llm_selector(ask: Ask, max_section_chars: int = 3000) -> LLMSelector:
    """Wrap ``ask(messages) -> reply text`` into a selector for ``PromptReducer.reduce(selector=...)``."""
    return LLMSelector(ask, max_section_chars)
