"""Task-based module dropping in the prompt reducer."""
from scre.blocks import parse_blocks
from scre.config import PromptConfig
from scre.prompt_modules import find_modules, tokens
from scre.prompt_reducer import PromptReducer
from scre.roles import classify

_FILLER = " ".join(["lorem"] * 120)  # >400 chars so a module is not "small"


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


def _modules(text, cfg=None):
    blocks = parse_blocks(text)
    classify(blocks)
    return find_modules(blocks, cfg or _cfg())


def test_sections_become_modules_and_wrapper_is_split():
    names = [m.name for m in _modules(_prompt())]
    assert names == ["safety", "gmail tool", "calendar tool", "weather notes"]


def test_no_task_only_removes_redundancy():
    r = PromptReducer(_cfg()).reduce(_prompt())
    assert r["context"] == _prompt().rstrip()


def test_task_drops_unmatched_sections_and_keeps_matched_and_global():
    r = PromptReducer(_cfg()).reduce(_prompt(), task="Check my gmail inbox for messages")
    out = r["context"]
    assert "## Gmail tool" in out and "gmail search endpoint" in out
    assert "## Safety" in out                      # global tier
    assert "Calendar tool" not in out and "barometer" not in out
    kept = {m["name"]: m["kept"] for m in r["metadata"]["modules"]}
    assert kept == {"safety": True, "gmail tool": True, "calendar tool": False, "weather notes": False}


def test_dropped_text_is_only_whole_original_blocks():
    text = _prompt()
    out = PromptReducer(_cfg()).reduce(text, task="gmail")["context"]
    assert all(line in text for line in out.splitlines())


def test_small_sections_are_never_dropped():
    text = "# A\n\n## Tiny\nUse the zebra endpoint.\n\n## Other\n" + _FILLER + "\n"
    out = PromptReducer(_cfg()).reduce(text, task="unrelated words here")["context"]
    assert "zebra" in out


def test_caller_drops_respect_guards():
    red = PromptReducer(_cfg())
    ids = {m["name"]: m["id"] for m in red.outline(_prompt())}
    r = red.reduce(_prompt(), task="gmail messages", drop_modules=[ids["safety"], ids["gmail tool"], ids["calendar tool"]])
    out = r["context"]
    assert "## Safety" in out        # global: guard overrides the caller
    assert "## Gmail tool" in out    # task mentions it: guard overrides the caller
    assert "Calendar tool" not in out
    assert "barometer" in out        # caller did not drop it


def test_dropping_inside_tag_containers_keeps_tags_balanced():
    text = (
        "<rules>\nBe nice.\n</rules>\n\n"
        f"<gmail>\nUse gmail. {_FILLER}\n</gmail>\n\n"
        f"<calendar>\nUse calendar. {_FILLER}\n</calendar>\n"
    )
    out = PromptReducer(_cfg()).reduce(text, task="gmail")["context"]
    assert "<gmail>" in out and "</gmail>" in out
    assert "calendar" not in out


def test_outline_lists_modules_with_preview():
    o = PromptReducer(_cfg()).outline(_prompt())
    assert [m["name"] for m in o][:2] == ["safety", "gmail tool"]
    assert o[1]["preview"].startswith("Use the gmail search") and o[0]["class"] == "global"


def test_tokens_lowercase_split_and_stem():
    assert tokens("Editing files, apply_patch") >= {"edit", "file", "apply", "patch"}


def _guard_prompt() -> str:
    return (
        "# Assistant\n\n"
        "## Working style\nTo change any file, call the `apply_patch` tool.\n\n"
        f"## apply_patch\nThe `apply_patch` tool takes a patch string. {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n"
    )


def test_reference_guard_keeps_a_section_the_kept_text_uses():
    red = PromptReducer(_cfg())
    ids = {m["name"]: m["id"] for m in red.outline(_guard_prompt())}
    r = red.reduce(_guard_prompt(), task="say hi", drop_modules=[ids["applypatch"], ids["weather notes"]])
    kept = {m["name"]: (m["kept"], m["reason"]) for m in r["metadata"]["modules"]}
    assert kept["applypatch"][0] and "apply_patch" in kept["applypatch"][1]
    assert not kept["weather notes"][0]
    assert "barometer" not in r["context"] and "takes a patch string" in r["context"]


def test_vocabulary_guard_keeps_the_section_whose_tags_the_examples_use():
    ex = "".join(f"<example>\n<reply>\n<answer id=\"{i}\">ok</answer>\n</reply>\n</example>\n\n" for i in range(3))
    text = (
        "# Assistant\n\n"
        f"## Answer format\nWrap every answer in the answer tag with an id. {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n\n"
        f"## Samples\n{ex}"
    )
    red = PromptReducer(_cfg(example_keep=2))
    ids = {m["name"]: m["id"] for m in red.outline(text)}
    r = red.reduce(text, task="unrelated", drop_modules=[ids["weather notes"]])
    assert "barometer" not in r["context"]
    # the examples use <answer>; the section that introduces the format must stay
    r2 = red.reduce(text, task="unrelated", drop_modules=[ids["answer format"]])
    assert "Wrap every answer" in r2["context"]


def test_outline_can_include_full_section_text():
    o = PromptReducer(_cfg()).outline(_prompt(), max_text_chars=60)
    assert o[1]["text"].startswith("Use the gmail search endpoint") and len(o[1]["text"]) <= 60


def _tools_prompt() -> str:
    return (
        "# Assistant\n\n"
        f"## Mail tool\nSend mail. namespace mail {{ type send = (_: {{to: string}}) => any; }} {_FILLER}\n\n"
        f"## Calendar tool\nBook meetings. namespace cal {{ type book = (_: {{when: string}}) => any; }} {_FILLER} {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n"
    )


def test_convention_guard_keeps_one_tool_signature_when_all_are_dropped():
    red = PromptReducer(_cfg())
    ids = {m["name"]: m["id"] for m in red.outline(_tools_prompt())}
    r = red.reduce(_tools_prompt(), task="unrelated", drop_modules=[ids["mail tool"], ids["calendar tool"], ids["weather notes"]])
    kept = {m["name"]: m["kept"] for m in r["metadata"]["modules"]}
    assert kept == {"mail tool": True, "calendar tool": False, "weather notes": False}   # the smaller signature stays
    assert "namespace mail" in r["context"] and "barometer" not in r["context"]


def test_convention_guard_does_nothing_when_a_signature_section_is_already_kept():
    red = PromptReducer(_cfg())
    ids = {m["name"]: m["id"] for m in red.outline(_tools_prompt())}
    r = red.reduce(_tools_prompt(), task="unrelated", drop_modules=[ids["mail tool"], ids["weather notes"]])
    kept = {m["name"]: m["kept"] for m in r["metadata"]["modules"]}
    assert kept["calendar tool"] and not kept["mail tool"]


def test_guards_cascade_to_a_fixpoint():
    # the only signature section that survives (mail) mentions `send_mail`, which the notes section defines
    text = (
        "# Assistant\n\n"
        f"## Mail tool\nSend mail with `send_mail`. namespace mail {{ type send = (_: {{to: string}}) => any; }} {_FILLER}\n\n"
        f"## Mail helper\nThe `send_mail` helper adds a signature. `send_mail` retries twice. {_FILLER} {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n"
    )
    red = PromptReducer(_cfg())
    ids = {m["name"]: m["id"] for m in red.outline(text)}
    r = red.reduce(text, task="unrelated", drop_modules=[ids["mail tool"], ids["mail helper"], ids["weather notes"]])
    kept = {m["name"]: m["kept"] for m in r["metadata"]["modules"]}
    assert kept == {"mail tool": True, "mail helper": True, "weather notes": False}
