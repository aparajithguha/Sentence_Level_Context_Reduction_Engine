"""
Unit tests for scre/units.py -- the SemanticUnit/WorkflowUnit data model
and CATEGORY_WEIGHTS. No spaCy pipeline or embedder needed.
"""
import json

from scre.units import SemanticUnit, WorkflowUnit, CATEGORY_WEIGHTS


def test_semantic_unit_defaults():
    u = SemanticUnit(0, "We chose PostgreSQL.", "decision")
    assert u.original_sentence_index == 0
    assert u.original_sentence_text == "We chose PostgreSQL."
    assert u.render_text == "We chose PostgreSQL."
    assert u.type == "decision"
    assert u.subject == "" and u.relation == "" and u.object == ""
    assert u.score == 0.0
    assert u.connectivity == 0
    assert u.context_header == ""
    assert u.entities == set()
    # New in this session: extraction provenance, defaults to "heuristic"
    # (the lower-confidence tier) unless an extractor explicitly upgrades
    # it -- see RegexKeyValueExtractor and WorkflowUnit.
    assert u.extraction_source == "heuristic"


def test_semantic_unit_type_is_lowercased():
    u = SemanticUnit(0, "text", "DECISION")
    assert u.type == "decision"


def test_semantic_unit_equality_ignores_index_and_text():
    a = SemanticUnit(0, "Sentence A.", "decision", subject="X", relation="chose", obj="Y")
    b = SemanticUnit(5, "Different wording entirely.", "decision", subject="x", relation="Chose", obj="y")
    assert a == b  # same type/subject/relation/object (case-insensitive)
    assert hash(a) == hash(b)


def test_semantic_unit_inequality_on_different_type():
    a = SemanticUnit(0, "text", "decision", subject="X", obj="Y")
    b = SemanticUnit(0, "text", "constraint", subject="X", obj="Y")
    assert a != b


def test_semantic_unit_eq_returns_notimplemented_for_other_types():
    u = SemanticUnit(0, "text", "decision")
    assert u.__eq__("not a unit") is NotImplemented


def test_workflow_unit_render_text_is_valid_json():
    wf = WorkflowUnit(0, "Deploy", ["1. Build", "2. Push", "3. Deploy"], "Deploy:\n1. Build\n2. Push\n3. Deploy")
    parsed = json.loads(wf.render_text)
    assert parsed["type"] == "workflow"
    assert parsed["name"] == "Deploy"
    assert parsed["steps"] == ["1. Build", "2. Push", "3. Deploy"]


def test_workflow_unit_type_is_workflow():
    wf = WorkflowUnit(0, "Deploy", ["1. Build"], "text")
    assert wf.type == "workflow"


def test_workflow_unit_extraction_source_is_labeled():
    # Structurally detected via header + list-item pattern -- ground truth,
    # same confidence tier as a RegexKeyValueExtractor match, not a guess.
    wf = WorkflowUnit(0, "Deploy", ["1. Build"], "text")
    assert wf.extraction_source == "labeled"


def test_workflow_unit_get_truncated_text():
    wf = WorkflowUnit(0, "Deploy", ["1. Build", "2. Push"], "text")
    truncated = wf.get_truncated_text()
    assert "Deploy" in truncated
    assert "2 steps" in truncated


def test_workflow_unit_equality_requires_matching_steps():
    a = WorkflowUnit(0, "Deploy", ["1. Build", "2. Push"], "text")
    b = WorkflowUnit(0, "Deploy", ["1. Build", "2. Push"], "text")
    c = WorkflowUnit(0, "Deploy", ["1. Build"], "text")
    assert a == b
    assert a != c


def test_category_weights_covers_core_types():
    for expected_type in ("constraint", "decision", "reason", "goal", "workflow", "task", "fact"):
        assert expected_type in CATEGORY_WEIGHTS
    # Weights are meant as relative importance signals, not raw booleans.
    assert all(0.0 < w <= 1.0 for w in CATEGORY_WEIGHTS.values())


def test_workflow_unit_render_text_keeps_non_ascii_readable():
    wf = WorkflowUnit(0, "Plan’s steps", ["1. Don’t stop – ever"], "text")
    assert "’" in wf.render_text and "–" in wf.render_text
    assert "\\u" not in wf.render_text
    assert json.loads(wf.render_text)["name"] == "Plan’s steps"
