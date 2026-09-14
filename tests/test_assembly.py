"""
Unit tests for scre/assembly.py -- final compressed-context rendering.
No spaCy pipeline or embedder needed; operates purely on SemanticUnit data.
"""
from scre.units import SemanticUnit
from scre.assembly import build_compressed_context


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
