"""
prompt_reducer.py
=================
``PromptReducer`` -- reduces a system/agent prompt while preserving every
instruction. Where ``SCRE.reduce`` selects the sentences of a document that
answer a query, this keeps the whole prompt and removes only what is
redundant: repeated instructions, surplus examples, decoration, and the
headings/tags left empty by those removals. It runs no model and rewrites no
text -- the output is the original prompt with whole blocks removed.

Pipeline: ``blocks.parse_blocks`` -> ``roles.classify`` ->
``prompt_graph.build_graph`` -> ``prompt_policy.decide`` ->
``prompt_assembly.assemble``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable

from .blocks import parse_blocks
from .config import PromptConfig
from .prompt_assembly import assemble
from .prompt_graph import build_graph
from .prompt_modules import find_modules, select_modules, symbol_counts
from .prompt_policy import decide
from .roles import classify
from .utils import estimate_tokens


class PromptReducer:
    def __init__(self, config: PromptConfig | None = None):
        self.config = config if config is not None else PromptConfig()

    def outline(self, text: str, max_text_chars: int = 0) -> list[dict[str, Any]]:
        """The task-selectable modules of ``text``: what a selector (a person,
        a heuristic, or an LLM) chooses from and passes back as ``drop_modules``.
        ``max_text_chars`` > 0 adds each section's own text (cut to that length)."""
        blocks = parse_blocks(text)
        classify(blocks)
        out = []
        for m in find_modules(blocks, self.config):
            row = {"id": m.id, "name": m.name, "path": list(m.path), "chars": m.chars, "class": m.cls, "preview": m.preview}
            if max_text_chars:
                row["text"] = m.body[:max_text_chars]
            out.append(row)
        return out

    def reduce(
        self, text: str, task: str | None = None, drop_modules: set[int] | list[int] | None = None,
        selector: Callable[[list[dict], str], Iterable[int]] | None = None,
    ) -> dict[str, Any]:
        """Reduce ``text``.

        With no ``task`` (and no ``drop_modules``) only redundancy is removed.
        With a ``task``, sections whose content the task cannot use are also
        dropped whole (see ``prompt_modules``). ``drop_modules`` -- ids from
        ``outline`` -- lets a caller choose the sections to drop instead; the
        global / small / task-mentioned guards still keep their sections.

        ``selector(outline, task) -> ids`` does that choosing for you (needs a
        ``task``; exclusive with ``drop_modules``). ``scre.prompt_select.llm_selector``
        builds one from your own LLM function. The selector's answer is treated
        like ``drop_modules``, so every guard still applies. If it raises,
        ``PromptConfig.on_selector_error`` decides: "keep" (redundancy-only) or
        "match" (the built-in task matcher); the error is in ``metadata["selector"]``.

        Returns:
            ``{"context": str, "metadata": {...}, "blocks": [...]}`` where
            ``blocks`` lists every block with its role and keep/drop decision.
        """
        blocks = parse_blocks(text)
        if not blocks:
            return {"context": text, "metadata": {"original_chars": len(text), "reduced_chars": len(text), "reduction_ratio": 0.0, "blocks": 0}, "blocks": []}
        classify(blocks)
        graph = build_graph(blocks, self.config)
        modules = find_modules(blocks, self.config)
        if selector is not None and drop_modules is not None:
            raise ValueError("pass either selector or drop_modules, not both")
        proposed = set(drop_modules) if drop_modules is not None else None
        selector_meta: dict[str, Any] | None = None
        if selector is not None:
            selector_meta = {"used": False, "error": None, "fallback": None}
            if task is None:
                selector_meta["error"] = "a selector needs a task; ran redundancy-only"
            else:
                try:
                    rows = self.outline(text, getattr(selector, "max_text_chars", self.config.selector_text_chars))
                    proposed = {int(i) for i in selector(rows, task)}
                    selector_meta["used"] = True
                    selector_meta["info"] = getattr(selector, "last", None)
                except Exception as e:  # noqa: BLE001 -- any selector failure must fall back, never break the prompt
                    selector_meta["error"] = f"{type(e).__name__}: {e}"[:200]
                    selector_meta["fallback"] = self.config.on_selector_error
                    if self.config.on_selector_error == "keep":
                        task = None
                        proposed = None
        module_calls = {}
        if task is not None or proposed is not None:
            in_modules = {i for m in modules for i in m.block_ids}
            outside = " ".join(b.text for b in blocks if b.id not in in_modules and b.role != "example")
            ex_use = Counter()
            for u in graph.example_units:
                ex_use.update(set(symbol_counts(" ".join(blocks[i].text for i in u.blocks))))
            module_calls = select_modules(modules, task, self.config, proposed, outside,
                                          frozenset(sym for sym, c in ex_use.items() if c >= 2))
        drop_blocks = {
            i: "module dropped"
            for m in modules if m.id in module_calls and not module_calls[m.id][0]
            for i in m.block_ids
        }
        keep, reason = decide(blocks, graph, self.config, drop_blocks)
        reduced = assemble(blocks, keep, self.config)

        dropped = Counter(r.split(" of block")[0] for r in reason.values())
        ex_units = [u for u in graph.example_units if not any(keep[i] for i in u.blocks)]
        ex_trimmed = [u for u in graph.example_units if any(reason.get(i) == "example trimmed" for i in u.blocks)]
        return {
            "context": reduced,
            "metadata": {
                "original_chars": len(text),
                "reduced_chars": len(reduced),
                "original_estimated_tokens": estimate_tokens(text),
                "reduced_estimated_tokens": estimate_tokens(reduced),
                "reduction_ratio": round(1 - len(reduced) / max(len(text), 1), 4),
                "blocks": len(blocks),
                "kept_blocks": sum(keep.values()),
                "dropped_by_reason": dict(dropped),
                "example_units": len(graph.example_units),
                "example_units_dropped": len(ex_units),          # every example that is gone, for any reason
                "example_units_trimmed": len(ex_trimmed),       # only those cut as surplus (over ``example_keep``)
                "example_units_dropped_with_rule_wording": sum(u.strong_rules for u in ex_units),
                "modules": [
                    {"id": m.id, "name": m.name, "chars": m.chars, "class": m.cls,
                     "kept": module_calls.get(m.id, (True, ""))[0], "reason": module_calls.get(m.id, (True, ""))[1]}
                    for m in modules
                ],
                **({"selector": selector_meta} if selector_meta is not None else {}),
            },
            "blocks": [
                {
                    "id": b.id, "kind": b.kind, "role": b.role, "kept": keep[b.id],
                    "reason": reason.get(b.id, ""), "chars": len(b.text), "text": b.text,
                }
                for b in blocks
            ],
        }
