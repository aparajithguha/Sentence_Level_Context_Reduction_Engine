"""
Unit tests for scre/blocks.py -- the lossless prompt structure reader.
No spaCy pipeline or embedder needed.
"""
from scre.blocks import parse_blocks, rebuild

MIXED = """You are a helpful agent.

# Rules

Your capabilities:
- Read files
- Write files

Then some prose.

<examples>
<example>
User: hi
</example>
</examples>

```python
# not a heading

x = 1
```

| a | b |
|---|---|
| 1 | 2 |

---
"""


def _kinds(text):
    return [b.kind for b in parse_blocks(text)]


def test_blocks_partition_the_input_exactly():
    for text in (MIXED, "", "   \n\n", "one line", "- a\n- b\n\n\n1. c\n", "<a>\nno close", "```\nunclosed fence\nstill code"):
        assert rebuild(parse_blocks(text)) == text


def test_fenced_code_is_one_atomic_block():
    blocks = [b for b in parse_blocks(MIXED) if b.kind == "code"]
    assert len(blocks) == 1
    assert "# not a heading" in blocks[0].text and "x = 1" in blocks[0].text
    assert not any(b.kind == "heading" and "not a heading" in b.name for b in parse_blocks(MIXED))


def test_unclosed_fence_runs_to_end_as_code():
    blocks = parse_blocks("intro\n\n```\ncode\n# still code\n")
    assert [b.kind for b in blocks] == ["paragraph", "code"]


def test_json_array_in_tags_is_one_block_and_brackets_in_strings_are_ignored():
    text = '<functions>\n[\n  {"name": "a", "description": "use ] and } carefully"},\n  {"name": "b"}\n]\n</functions>\n'
    blocks = parse_blocks(text)
    assert [b.kind for b in blocks] == ["tag_open", "json", "tag_close"]
    assert blocks[1].parent == blocks[0].id
    assert blocks[0].pair == blocks[2].id


def test_bracketed_prose_is_not_json():
    assert _kinds("[User Info]\n") == ["paragraph"]
    assert _kinds("{{ template_var }}\n") == ["paragraph"]


def test_tag_without_closer_is_plain_text():
    assert _kinds("<orphan>\ntext\n") == ["paragraph"]


def test_nested_tags_pair_correctly():
    blocks = parse_blocks("<outer>\n<inner>\nx\n</inner>\n</outer>\n")
    kinds = [(b.kind, b.name) for b in blocks]
    assert kinds == [("tag_open", "outer"), ("tag_open", "inner"), ("paragraph", ""), ("tag_close", "inner"), ("tag_close", "outer")]
    assert blocks[0].pair == 4 and blocks[1].pair == 3
    assert blocks[2].parent == blocks[1].id and blocks[1].parent == blocks[0].id


def test_headings_nest_by_level():
    blocks = parse_blocks("# A\n\n## B\n\ntext\n\n# C\n\nmore\n")
    by = {b.name or b.text.strip(): b for b in blocks}
    assert by["b"].parent == by["a"].id
    assert by["text"].parent == by["b"].id
    assert by["c"].parent is None and by["more"].parent == by["c"].id


def test_html_named_tags_are_markup_not_containers():
    assert _kinds("<a>\nx\n</a>\n") == ["markup"]


def test_html_block_is_atomic_markup():
    blocks = parse_blocks("Intro\n\n<html>\n<body>\n<p>hi</p>\n</body>\n</html>\n\nAfter\n")
    assert [b.kind for b in blocks] == ["paragraph", "markup", "paragraph"]


def test_list_runs_and_ordering():
    blocks = parse_blocks("- a\n- b\n\n1. one\n2. two\n")
    items = [b for b in blocks if b.kind == "item"]
    assert [b.ordered for b in items] == [False, False, True, True]
    assert items[0].list_id == items[1].list_id == items[0].id
    assert items[2].list_id == items[3].list_id == items[2].id


def test_label_introduces_its_list_but_not_the_paragraph_after():
    blocks = parse_blocks("Your capabilities:\n- a\n- b\n\nAfter the list.\n")
    assert [b.kind for b in blocks] == ["label", "item", "item", "paragraph"]
    assert blocks[1].label == blocks[2].label == "your capabilities"
    assert blocks[3].label == ""


def test_item_continuation_lines_stay_in_the_item():
    blocks = parse_blocks("- first line\n  continued here\n- second\n")
    assert [b.kind for b in blocks] == ["item", "item"]
    assert "continued here" in blocks[0].text
