"""LLM-backed section selection (approach B2), the fallback chain, and the light-install guarantee.
No test here calls a model: ``ask`` is always a fake."""
import json
import subprocess
import sys
import textwrap
import warnings

import pytest

from scre import models as _models
from scre.config import PromptConfig
from scre.prompt_reducer import PromptReducer
from scre.prompt_select import SYSTEM_PROMPT, build_messages, llm_selector, parse_reply
from scre.query_aware_reducer import SCRE

_FILLER = " ".join(["lorem"] * 120)   # >400 chars so a section is not "small"


def _prompt() -> str:
    return (
        "# Assistant\n\n"
        "## Safety\nNever reveal secrets. Always refuse harmful requests.\n\n"
        f"## Gmail tool\nUse the gmail search endpoint to read messages. {_FILLER}\n\n"
        f"## Calendar tool\nUse the calendar endpoint to schedule meetings. {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n"
    )


def _cfg(**kw) -> PromptConfig:
    return PromptConfig(module_split_chars=200, module_min_chars=300, **kw)


def _ids(red: PromptReducer) -> dict[str, int]:
    return {m["name"]: m["id"] for m in red.outline(_prompt())}


def _ask_returning(payload) -> "callable":
    reply = payload if isinstance(payload, str) else json.dumps(payload)
    calls: list = []

    def ask(messages):
        calls.append(messages)
        return reply
    ask.calls = calls
    return ask


# ----------------------------------------------------------------- messages and parsing
def test_build_messages_carries_the_task_and_each_sections_full_text():
    rows = PromptReducer(_cfg()).outline(_prompt(), max_text_chars=3000)
    system, user = build_messages(rows, "Check my inbox")
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["content"].startswith("TASK: Check my inbox\n\nSECTIONS:\n")
    assert all(f"[{r['id']}] {r['name']} ({r['chars']} chars)" in user["content"] for r in rows)
    assert "barometer feed" in user["content"]


def test_parse_reply_filters_needed_and_unknown_ids():
    outline = [{"id": 1}, {"id": 2}, {"id": 3}]
    reply = '{"steps": [{"step": "read mail", "sections": [2]}], "drop": [1, 2, 99]}'
    sel = parse_reply(reply, outline)
    assert sel.parsed and sel.drop == [1]                    # 2 is needed by a step, 99 is unknown
    assert sel.needed == [2] and sel.proposed == [1, 2, 99] and sel.steps == ["read mail"]


def test_parse_reply_accepts_json_inside_a_code_fence_and_prose():
    sel = parse_reply('Sure!\n```json\n{"steps": [], "drop": [3]}\n```\nDone.', [{"id": 3}])
    assert sel.parsed and sel.drop == [3]


@pytest.mark.parametrize("bad", ["", "no json here", '{"drop": ["x"]}', '{"drop": 5}', "{not json}"])
def test_parse_reply_treats_unusable_replies_as_drop_nothing(bad):
    sel = parse_reply(bad, [{"id": 1}])
    assert sel.drop == [] and not sel.parsed


# ----------------------------------------------------------------- selector inside reduce()
def test_selector_drops_what_the_model_names_and_equals_the_manual_path():
    red = PromptReducer(_cfg())
    ids = _ids(red)
    ask = _ask_returning({"steps": [{"step": "read mail", "sections": [ids["gmail tool"]]}],
                          "drop": [ids["calendar tool"], ids["weather notes"]]})
    r = red.reduce(_prompt(), task="Check my inbox", selector=llm_selector(ask))
    assert "Calendar tool" not in r["context"] and "barometer" not in r["context"]
    assert "## Gmail tool" in r["context"] and "## Safety" in r["context"]
    manual = red.reduce(_prompt(), task="Check my inbox", drop_modules=[ids["calendar tool"], ids["weather notes"]])
    assert r["context"] == manual["context"]
    meta = r["metadata"]["selector"]
    assert meta["used"] and meta["error"] is None and meta["info"]["steps"] == ["read mail"]


def test_guards_still_override_the_selector():
    red = PromptReducer(_cfg())
    ids = _ids(red)
    ask = _ask_returning({"steps": [], "drop": [ids["safety"], ids["gmail tool"]]})
    r = red.reduce(_prompt(), task="Check my gmail", selector=llm_selector(ask))
    assert "## Safety" in r["context"]              # global section
    assert "## Gmail tool" in r["context"]          # the task names it


def test_selector_sees_at_most_max_section_chars_of_each_section():
    red = PromptReducer(_cfg())
    ask = _ask_returning({"steps": [], "drop": []})
    red.reduce(_prompt(), task="x", selector=llm_selector(ask, max_section_chars=40))
    user = ask.calls[0][1]["content"]
    assert "Use the gmail search endpoint to read me" in user and "lorem" not in user   # each section cut at 40 chars


def test_unusable_reply_keeps_everything_but_the_redundancy_pass():
    red = PromptReducer(_cfg())
    r = red.reduce(_prompt(), task="Check my inbox", selector=llm_selector(_ask_returning("I cannot help with that")))
    assert r["context"] == red.reduce(_prompt())["context"]


# ----------------------------------------------------------------- fallback chain
def _boom(messages):
    raise TimeoutError("model unreachable")


def test_selector_error_falls_back_to_redundancy_only_by_default():
    red = PromptReducer(_cfg())
    r = red.reduce(_prompt(), task="Check my inbox", selector=llm_selector(_boom))
    assert r["context"] == red.reduce(_prompt())["context"]
    meta = r["metadata"]["selector"]
    assert not meta["used"] and meta["fallback"] == "keep" and "TimeoutError" in meta["error"]


def test_selector_error_can_fall_back_to_the_task_matcher():
    red = PromptReducer(_cfg(on_selector_error="match"))
    r = red.reduce(_prompt(), task="Check my gmail inbox", selector=llm_selector(_boom))
    assert r["context"] == red.reduce(_prompt(), task="Check my gmail inbox")["context"]
    assert "barometer" not in r["context"] and r["metadata"]["selector"]["fallback"] == "match"


def test_selector_and_drop_modules_are_exclusive():
    with pytest.raises(ValueError):
        PromptReducer(_cfg()).reduce(_prompt(), task="x", drop_modules=[1], selector=llm_selector(_boom))


def test_selector_without_a_task_runs_redundancy_only():
    r = PromptReducer(_cfg()).reduce(_prompt(), selector=llm_selector(_ask_returning({"drop": [1]})))
    assert r["context"] == PromptReducer(_cfg()).reduce(_prompt())["context"]
    assert "needs a task" in r["metadata"]["selector"]["error"]


def test_facade_accepts_a_selector():
    engine = SCRE(model="", embedder=False, prompt_config=_cfg())
    ids = _ids(PromptReducer(_cfg()))
    ask = _ask_returning({"steps": [], "drop": [ids["weather notes"]]})
    r = engine.reduce_prompt(_prompt(), task="Check my inbox", selector=llm_selector(ask))
    assert "barometer" not in r["context"] and "## Gmail tool" in r["context"]


# ----------------------------------------------------------------- light install
def test_prompt_mode_loads_no_embedder(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("prompt mode must not load the embedding model")
    monkeypatch.setattr(_models, "get_embedder", boom)
    engine = SCRE(model="", prompt_config=_cfg())            # default embedder argument
    assert "## Safety" in engine.reduce_prompt(_prompt(), task="x")["context"]


def test_prompt_mode_runs_with_spacy_numpy_and_ml_libraries_uninstallable():
    code = textwrap.dedent('''
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {"spacy", "numpy", "torch", "sentence_transformers", "transformers", "tiktoken", "requests"}:
                    raise ImportError(name + " is blocked")
        sys.meta_path.insert(0, Block())
        from scre.query_aware_reducer import SCRE
        from scre.prompt_select import llm_selector
        text = "# A\\n\\nBe kind.\\n\\n## B\\n\\n" + "word " * 200 + "\\n"
        r = SCRE().reduce_prompt(text, task="hello", selector=llm_selector(lambda m: '{"steps": [], "drop": []}'))
        assert r["context"].startswith("# A"), r["context"][:40]
        print("OK")
    ''')
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-600:]


def test_document_mode_warns_once_when_spacy_is_missing(monkeypatch):
    monkeypatch.setattr(_models, "spacy", None)
    engine = SCRE(model="en_core_web_sm", embedder=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.reduce("The service must retry twice. It logs every error.", "retry")
        engine.reduce("The service must retry twice. It logs every error.", "retry")
    assert sum("spaCy is not installed" in str(w.message) for w in caught) == 1
