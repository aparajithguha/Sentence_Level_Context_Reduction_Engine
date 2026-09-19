"""
prompt_adapter.py
=================
Consumer-side wrapper for the demo UI's "System prompt" mode. Like ``reducer_adapter``, it only uses
``scre``'s public surface and never edits ``scre/``.

It runs ``PromptReducer`` and reshapes the result for the page: per-section keep/drop with the reason,
what else was removed, and -- when an LLM chose the sections -- the model's plan. The LLM is the
demo's local Ollama model; the library itself never calls one (``scre.prompt_select.llm_selector``
just wraps whatever ``ask`` function it is given).
"""
from __future__ import annotations

from typing import Any, Callable

from scre.prompt_reducer import PromptReducer
from scre.prompt_select import llm_selector

from showcase.reducer_adapter import MODEL_NAMES

STRATEGIES = ("off", "match", "llm")
# Keep the selector's input inside a 16K-token context on a laptop: the more sections, the less of each is shown.
SELECTOR_INPUT_CHARS = 45_000
NUM_CTX = 16_384


def ollama_ask(model_name: str) -> Callable[[list[dict]], str]:
    """An ``ask(messages) -> reply text`` backed by a local Ollama model."""
    def ask(messages: list[dict]) -> str:
        import ollama
        response = ollama.chat(model=model_name, messages=messages, format="json",
                               options={"num_ctx": NUM_CTX, "temperature": 0})
        return response["message"]["content"]
    return ask


class PromptExplainer:
    def __init__(self, reducer: PromptReducer | None = None):
        self.reducer = reducer if reducer is not None else PromptReducer()

    def reduce(
        self, text: str, task: str = "", strategy: str = "off", model_key: str = "qwen3",
        ask: Callable[[list[dict]], str] | None = None,
    ) -> dict[str, Any]:
        """Reduce ``text`` as a system prompt.

        ``strategy``: ``"off"`` (redundancy only), ``"match"`` (built-in word matching against ``task``) or
        ``"llm"`` (an LLM chooses sections for ``task``). ``ask`` overrides the default Ollama model, for tests.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}'")
        task = (task or "").strip()
        notice = None
        effective = strategy
        if strategy != "off" and not task:
            effective, notice = "off", "Add a task to drop sections it does not need; ran redundancy-only."

        selector = None
        if effective == "match":
            result = self.reducer.reduce(text, task=task)
        elif effective == "llm":
            n_sections = len(self.reducer.outline(text))
            per_section = min(3000, max(300, SELECTOR_INPUT_CHARS // max(n_sections, 1)))
            selector = llm_selector(ask or ollama_ask(MODEL_NAMES[model_key]), max_section_chars=per_section)
            result = self.reducer.reduce(text, task=task, selector=selector)
        else:
            result = self.reducer.reduce(text)

        meta = result["metadata"]
        sel_meta = meta.get("selector")
        info = (sel_meta or {}).get("info") or {}
        if sel_meta and sel_meta.get("error"):
            notice = f"The model could not be used ({sel_meta['error']}); ran redundancy-only."
        elif sel_meta and sel_meta.get("used") and info and not info.get("parsed", True):
            notice = "The model's reply could not be read, so no sections were dropped."

        removed = {k: v for k, v in meta["dropped_by_reason"].items() if k != "module dropped"}
        # The library counts dropped blocks; the page should count what a reader sees: examples, and tag pairs.
        if "example trimmed" in removed:
            removed["example trimmed"] = meta["example_units_trimmed"]
        if "empty container" in removed:
            removed["empty container"] = max(1, removed["empty container"] // 2)

        sections = [
            {"name": m["name"], "chars": m["chars"], "class": m["class"], "kept": m["kept"], "reason": m["reason"]}
            for m in meta["modules"]
        ]
        return {
            "mode": "system_prompt",
            "strategy": effective,
            "context": result["context"],
            "metadata": {k: meta[k] for k in ("original_chars", "reduced_chars", "original_estimated_tokens",
                                              "reduced_estimated_tokens", "reduction_ratio")}
                        | {"sections_total": len(sections), "sections_kept": sum(s["kept"] for s in sections)},
            "sections": sections,
            "removed_by_reason": removed,
            "plan": info.get("steps", []) if sel_meta and sel_meta.get("used") else [],
            "notice": notice,
        }
