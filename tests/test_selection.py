"""
Unit tests for scre/selection.py -- token-budgeted selection and
post-selection expansion. No spaCy pipeline or embedder needed; operates
purely on hand-built, pre-scored SemanticUnit data.
"""
from scre.units import SemanticUnit
from scre.config import ScoringConfig
from scre.selection import select_semantic_units, expand_adjacent_context, expand_reasoning_chains


def _scored(index, text, type_="fact", score=1.0):
    # subject/object include the index so units are never spuriously equal
    # under SemanticUnit.__eq__ (which compares type/subject/relation/object,
    # not original_sentence_index) -- distinct sentences must stay distinct.
    u = SemanticUnit(index, text, type_, subject=f"subj{index}", obj=f"obj{index}")
    u.score = score
    return u


# --- select_semantic_units ---

def test_select_semantic_units_filters_nonpositive_scores():
    units = [_scored(0, "a", score=1.0), _scored(1, "b", score=0.0), _scored(2, "c", score=-1.0)]
    selected = select_semantic_units(units=units, max_sentences=5, max_tokens=None, min_tokens=0)
    assert len(selected) == 1
    assert selected[0].original_sentence_index == 0


def test_select_semantic_units_respects_max_sentences_without_reasoning_graph():
    units = [_scored(i, f"distinct sentence number {i} with unique content", score=float(10 - i)) for i in range(6)]
    selected = select_semantic_units(units=units, max_sentences=2, max_tokens=None, min_tokens=0)
    assert len(selected) == 2
    # Highest-scored two must win.
    assert {u.original_sentence_index for u in selected} == {0, 1}


def test_select_semantic_units_dedups_near_duplicates_via_jaccard():
    cfg = ScoringConfig(jaccard_dedup_threshold=0.85)
    a = _scored(0, "the quick brown fox jumps over the lazy dog", score=10.0)
    near_dup = _scored(1, "the quick brown fox jumps over the lazy dog today", score=9.0)
    distinct = _scored(2, "completely unrelated content about databases", score=8.0)
    selected = select_semantic_units(units=[a, near_dup, distinct], max_sentences=5, max_tokens=None, min_tokens=0, config=cfg)
    indices = {u.original_sentence_index for u in selected}
    assert 0 in indices
    assert 1 not in indices  # deduped against 0
    assert 2 in indices


def test_select_semantic_units_min_tokens_floor_extends_past_max_sentences():
    # Short, distinct units mean max_sentences alone wouldn't reach
    # min_tokens -- the floor should pull in additional positively-scored
    # content. Text must differ per unit or Jaccard dedup collapses them.
    units = [_scored(i, f"short text {i}", score=float(10 - i)) for i in range(6)]
    selected_no_floor = select_semantic_units(units=units, max_sentences=1, max_tokens=None, min_tokens=0)
    selected_with_floor = select_semantic_units(units=units, max_sentences=1, max_tokens=None, min_tokens=1000)
    assert len(selected_with_floor) > len(selected_no_floor)


def test_select_semantic_units_floor_extension_capped_by_ceiling_multiplier():
    cfg = ScoringConfig(selection_ceiling_multiplier=2)
    units = [_scored(i, "x", score=float(20 - i)) for i in range(10)]
    selected = select_semantic_units(units=units, max_sentences=2, max_tokens=None, min_tokens=100000, config=cfg)
    # Floor may pad past max_sentences, but never past ceiling_multiplier * max_sentences.
    assert len(selected) <= 2 * cfg.selection_ceiling_multiplier


def test_select_semantic_units_reasoning_graph_tie_break_prefers_connected():
    # Regression pin: among equally-scored candidates, once the base
    # max_sentences budget is filled, extension must prefer candidates
    # connected to the top-ranked unit over unconnected ones that merely
    # appear earlier in the document.
    top = _scored(0, "top ranked unit", score=10.0)
    connected = _scored(1, "connected to top", score=5.0)
    unconnected_earlier = _scored(2, "unconnected but would win by document order", score=5.0)
    reasoning_graph = {0: {1}, 1: {0}, 2: set()}

    selected = select_semantic_units(
        units=[top, connected, unconnected_earlier],
        max_sentences=1,
        max_tokens=None,
        min_tokens=100000,  # force the floor to try extending past max_sentences
        reasoning_graph=reasoning_graph,
    )
    indices = {u.original_sentence_index for u in selected}
    assert 0 in indices
    assert 1 in indices  # connected candidate extends the selection
    assert 2 not in indices  # unconnected candidate is gated out despite equal score


# --- expand_adjacent_context ---

def test_expand_adjacent_context_includes_window_on_both_sides():
    all_units = [_scored(i, f"sentence {i}") for i in range(5)]
    selected = [all_units[2]]
    expanded = expand_adjacent_context(selected, all_units, window=1)
    assert {u.original_sentence_index for u in expanded} == {1, 2, 3}


def test_expand_adjacent_context_zero_window_is_noop():
    all_units = [_scored(i, f"sentence {i}") for i in range(3)]
    selected = [all_units[1]]
    expanded = expand_adjacent_context(selected, all_units, window=0)
    assert {u.original_sentence_index for u in expanded} == {1}


def test_expand_adjacent_context_returns_sorted_by_index():
    all_units = [_scored(i, f"sentence {i}") for i in range(5)]
    selected = [all_units[3], all_units[0]]
    expanded = expand_adjacent_context(selected, all_units, window=0)
    assert [u.original_sentence_index for u in expanded] == [0, 3]


# --- expand_reasoning_chains ---

def test_expand_reasoning_chains_pulls_in_two_hop_neighbors():
    all_units = [_scored(i, f"sentence {i}") for i in range(4)]
    reasoning_graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    selected = [all_units[0]]
    expanded = expand_reasoning_chains(selected, all_units, reasoning_graph)
    # 0 -> hop1 {1} -> hop2 {0, 2} => {0, 1, 2}
    assert {u.original_sentence_index for u in expanded} == {0, 1, 2}


def test_expand_reasoning_chains_with_no_edges_returns_selection_unchanged():
    all_units = [_scored(i, f"sentence {i}") for i in range(3)]
    selected = [all_units[1]]
    expanded = expand_reasoning_chains(selected, all_units, reasoning_graph={0: set(), 1: set(), 2: set()})
    assert {u.original_sentence_index for u in expanded} == {1}


# --- expansion must be by position, not structural equality ---

def _bare(index, text):
    # Deliberately identical type/subject/relation/object across positions
    # (as e.g. markdown headings are): SemanticUnit.__eq__ treats these as equal.
    u = SemanticUnit(index, text, "fact")
    u.score = 1.0
    return u


def test_expand_adjacent_context_keeps_structurally_equal_units_at_other_positions():
    units = [_bare(0, "a"), _bare(1, "b"), _bare(2, "c")]
    assert units[0] == units[1] == units[2]
    out = expand_adjacent_context([units[1]], units, window=1)
    assert [u.original_sentence_index for u in out] == [0, 1, 2]


def test_expand_reasoning_chains_keeps_structurally_equal_units_at_other_positions():
    units = [_bare(0, "a"), _bare(1, "b"), _bare(2, "c")]
    graph = {0: {1}, 1: {0, 2}, 2: {1}}
    out = expand_reasoning_chains([units[1]], units, graph)
    assert [u.original_sentence_index for u in out] == [0, 1, 2]
