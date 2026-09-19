"""
Unit tests for scre/assembly.py -- final compressed-context rendering.
No spaCy pipeline or embedder needed; operates purely on SemanticUnit data.
"""
from scre.units import SemanticUnit
from scre.assembly import build_compressed_context, drop_orphan_headings


def test_build_compressed_context_empty_input():
    assert build_compressed_context([]) == ""


def test_build_compressed_context_orders_by_sentence_index():
    a = SemanticUnit(2, "Third sentence.", "fact")
    b = SemanticUnit(0, "First sentence.", "fact")
    c = SemanticUnit(1, "Second sentence.", "fact")
    result = build_compressed_context([a, b, c])
    lines = result.split("\n")
    assert lines == ["First sentence.", "Second sentence.", "Third sentence."]


def test_build_compressed_context_dedups_same_sentence_index():
    # Two units extracted from the same sentence (e.g. overlapping triples)
    # must not duplicate that sentence in the final output.
    a = SemanticUnit(0, "We chose PostgreSQL.", "decision", subject="we", obj="postgresql")
    b = SemanticUnit(0, "We chose PostgreSQL.", "outcome", subject="we", obj="reliability")
    result = build_compressed_context([a, b])
    assert result.count("We chose PostgreSQL.") == 1


def test_build_compressed_context_uses_render_text_not_original_text():
    u = SemanticUnit(0, "raw original text", "fact")
    u.render_text = "overridden render text"
    assert build_compressed_context([u]) == "overridden render text"


def test_build_compressed_context_injects_header_on_change():
    a = SemanticUnit(0, "First point.", "fact")
    a.context_header = "Section A"
    b = SemanticUnit(1, "Second point.", "fact")
    b.context_header = "Section A"
    c = SemanticUnit(2, "Third point.", "fact")
    c.context_header = "Section B"

    result = build_compressed_context([a, b, c])
    assert "[Section A]" in result
    assert "[Section B]" in result
    # Header injected only once per contiguous run of the same section --
    # not repeated for every unit within it.
    assert result.count("[Section A]") == 1


def test_build_compressed_context_no_header_when_original_text_starts_with_hash():
    # A unit whose own text is a Markdown heading shouldn't get a
    # redundant [header] prefix injected in front of itself.
    u = SemanticUnit(0, "# Section A", "fact")
    u.context_header = "Section A"
    result = build_compressed_context([u])
    assert result == "# Section A"


# ---- stray and empty headings -----------------------------------------------------------------
def _texts(units):
    # SemanticUnit equality is structural (type/subject/relation/object), so compare texts, not units.
    return [u.original_sentence_text for u in units]


def _u(i, text, header="Doc"):
    u = SemanticUnit(i, text, "fact")
    u.context_header = header
    return u


def test_a_heading_with_nothing_selected_under_it_is_dropped():
    units = [_u(0, "## Goals"), _u(1, "Body under goals."), _u(2, "## Risks")]
    assert build_compressed_context(units) == "[Doc]\n## Goals\nBody under goals."


def test_a_heading_followed_by_a_heading_of_the_same_level_is_dropped():
    units = [_u(0, "## A"), _u(1, "## B"), _u(2, "Body of B.")]
    assert _texts(drop_orphan_headings(units)) == ["## B", "Body of B."]


def test_a_parent_heading_stays_when_a_subsection_under_it_has_a_body():
    units = [_u(0, "## A"), _u(1, "### A1"), _u(2, "### A2"), _u(3, "Body of A2."), _u(4, "## B"), _u(5, "## C")]
    assert _texts(drop_orphan_headings(units)) == ["## A", "### A2", "Body of A2."]   # A1 is empty; B and C have nothing under them


def test_a_body_from_a_later_section_does_not_keep_an_earlier_heading():
    everything = [_u(0, "## A"), _u(1, "Body of A."), _u(2, "## B"), _u(3, "Body of B.")]
    selected = [everything[0], everything[3]]                # A's own body was not selected
    assert _texts(drop_orphan_headings(selected, everything)) == ["Body of B."]
    # Without the full document only the selected units are known, so it errs on the side of keeping.
    assert _texts(drop_orphan_headings(selected)) == ["## A", "Body of B."]


def test_if_only_headings_are_selected_they_are_all_kept():
    assert _texts(drop_orphan_headings([_u(0, "## A"), _u(1, "## B")])) == ["## A", "## B"]


def test_dropping_orphan_headings_is_idempotent():
    units = [_u(0, "## A"), _u(1, "Body."), _u(2, "## B")]
    once = drop_orphan_headings(units)
    assert _texts(once) == ["## A", "Body."] and _texts(drop_orphan_headings(once)) == _texts(once)


def test_a_multi_line_unit_that_starts_with_a_hash_counts_as_body_not_as_a_heading():
    workflow = _u(1, "## Plan\n1. Do this\n2. Do that")     # a workflow unit carries its header and its steps
    assert _texts(drop_orphan_headings([_u(0, "## A"), workflow])) == ["## A", "## Plan\n1. Do this\n2. Do that"]


def test_the_title_prefix_comes_before_a_leading_subheading():
    result = build_compressed_context([_u(0, "## Goals", header="Title"), _u(1, "Body.", header="Title")])
    assert result == "[Title]\n## Goals\nBody."


def test_the_h1_title_line_gets_no_prefix_and_neither_do_the_units_under_it():
    result = build_compressed_context([_u(0, "# Title", header="Title"), _u(1, "Body.", header="Title")])
    assert result == "# Title\nBody."
