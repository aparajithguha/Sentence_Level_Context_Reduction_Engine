"""
Unit tests for scre/extraction.py -- sentence segmentation and the
extractor chain. Uses nlp=None throughout (the regex-only fallback path)
so these tests need no spaCy pipeline -- DefaultNLPExtractor specifically
requires real spaCy dependency parsing and is exercised instead by the
existing integration tests in test_correctness.py.
"""
from scre.extraction import (
    RegexKeyValueExtractor,
    UnitExtractorStrategy,
    segment_text,
    build_semantic_state,
)
from scre.units import SemanticUnit


# --- segment_text (nlp=None regex fallback) ---

def test_segment_text_splits_plain_prose_into_sentences():
    sentences = segment_text("First sentence. Second sentence.", nlp=None)
    texts = [s.text for s in sentences]
    assert len(texts) == 2
    assert "First sentence." in texts[0]


def test_segment_text_keeps_list_items_intact_as_single_sentences():
    text = "Steps:\n1. Build the image\n2. Push the image"
    sentences = segment_text(text, nlp=None)
    texts = [s.text.strip() for s in sentences]
    assert "1. Build the image" in texts
    assert "2. Push the image" in texts


def test_segment_text_merges_split_label_value_pairs():
    # "Label:" on one line, value on the next -- must be merged into a
    # single structured line, not left as two disconnected sentences.
    text = "Decision:\nUse PostgreSQL."
    sentences = segment_text(text, nlp=None)
    texts = [s.text.strip() for s in sentences]
    assert any("Decision: Use PostgreSQL." in t for t in texts)


def test_segment_text_keeps_structured_multiline_paragraph_lines_atomic():
    # Regression pin (Bug 1): a multi-line paragraph mixing structured
    # Label: value lines must segment each line as its own atomic sentence,
    # not fuse "Decision:" with the adjacent "Reason:" line.
    text = "Decision: Use PostgreSQL.\nReason: Strong transactional guarantees."
    sentences = segment_text(text, nlp=None)
    texts = [s.text.strip() for s in sentences]
    assert "Decision: Use PostgreSQL." in texts
    assert "Reason: Strong transactional guarantees." in texts


def test_segment_text_does_not_force_atomic_on_single_line_prose():
    # Regression pin (Bug 5): a single-line paragraph starting with a word
    # and a colon must still segment normally into separate sentences, not
    # get wrongly frozen as one atomic unit.
    text = "Note: We should consider caching. This reduces latency significantly."
    sentences = segment_text(text, nlp=None)
    assert len(sentences) == 2


def test_segment_text_empty_string_returns_no_sentences():
    assert segment_text("", nlp=None) == []


def test_segment_text_does_not_force_atomic_on_single_line_starting_with_bullet():
    # Regression pin: a document with no blank lines at all (e.g. a system
    # prompt pasted as one unbroken block) whose text happens to start with
    # a bullet/dash character was previously treated as a single list item
    # and left completely unsplit, however long it was -- is_list_item had
    # no "more than one line" guard, unlike is_structured_line (Bug 5
    # above). A single `\n`-free line starting with "- " must still
    # segment normally into separate sentences.
    text = "- We should consider caching. This reduces latency significantly."
    sentences = segment_text(text, nlp=None)
    assert len(sentences) == 2


# --- RegexKeyValueExtractor ---

class _Sent:
    def __init__(self, text):
        self.text = text


def test_regex_key_value_extractor_markdown_format():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    units = extractor.extract(0, _Sent("Decision: Use PostgreSQL."))
    assert len(units) == 1
    assert units[0].type == "decision"
    assert units[0].object == "Use PostgreSQL."


def test_regex_key_value_extractor_tags_extraction_source_as_labeled():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    units = extractor.extract(0, _Sent("Constraint: Do not expose data."))
    assert units[0].extraction_source == "labeled"


def test_regex_key_value_extractor_sdlc_format_uses_type_map():
    extractor = RegexKeyValueExtractor(
        r"^(?:-\s*)?\[([A-Z]+)-\d+\]\s*([^:]+):\s*(.*)",
        {"DEC": "decision", "CON": "constraint"},
    )
    units = extractor.extract(0, _Sent("[DEC-01] Database: Use PostgreSQL."))
    assert len(units) == 1
    assert units[0].type == "decision"
    assert units[0].subject == "Database"
    assert units[0].object == "Use PostgreSQL."


def test_regex_key_value_extractor_canonicalizes_requirement_to_constraint():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    units = extractor.extract(0, _Sent("Requirement: Maintain explainability."))
    assert units[0].type == "constraint"


def test_regex_key_value_extractor_falls_back_to_fact_for_unknown_label():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    units = extractor.extract(0, _Sent("RandomLabel: some value."))
    assert units[0].type == "fact"


def test_regex_key_value_extractor_no_match_returns_empty_list():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    assert extractor.extract(0, _Sent("plain sentence with no colon structure")) == []


def test_regex_key_value_extractor_empty_value_returns_no_unit():
    extractor = RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
    assert extractor.extract(0, _Sent("Decision:")) == []


# --- build_semantic_state ---

def test_build_semantic_state_routes_through_extractor_chain():
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    sentences = segment_text("Decision: Use PostgreSQL.\nReason: Reliability.", nlp=None)
    units = build_semantic_state(sentences, extractors)
    types = [u.type for u in units]
    assert "decision" in types
    assert "reason" in types


def test_build_semantic_state_detects_named_workflow():
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    text = "Deployment Steps:\n1. Build the image\n2. Push the image\n3. Deploy"
    sentences = segment_text(text, nlp=None)
    units = build_semantic_state(sentences, extractors)
    workflows = [u for u in units if u.type == "workflow"]
    assert len(workflows) == 1
    assert len(workflows[0].steps) == 3
    assert workflows[0].extraction_source == "labeled"


def test_build_semantic_state_detects_anonymous_workflow():
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    text = "1. First step\n2. Second step"
    sentences = segment_text(text, nlp=None)
    units = build_semantic_state(sentences, extractors)
    workflows = [u for u in units if u.type == "workflow"]
    assert len(workflows) == 1
    assert workflows[0].name == "Unnamed Workflow"


def test_build_semantic_state_tracks_markdown_heading_as_context_header():
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    text = "# Section One\nDecision: Use PostgreSQL."
    sentences = segment_text(text, nlp=None)
    units = build_semantic_state(sentences, extractors)
    decisions = [u for u in units if u.type == "decision"]
    assert decisions[0].context_header == "Section One"


def test_build_semantic_state_stamps_entity_less_units_with_empty_set():
    # DummySentence (the nlp=None path) has no .ents -- entities must
    # degrade to an empty set, not raise.
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    sentences = segment_text("Decision: Use PostgreSQL.", nlp=None)
    units = build_semantic_state(sentences, extractors)
    assert units[0].entities == set()


def test_build_semantic_state_unmatched_sentence_yields_no_units():
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")]
    sentences = segment_text("Just a plain sentence with no label structure.", nlp=None)
    units = build_semantic_state(sentences, extractors)
    assert units == []


# --- bullets are not workflows; numbered lists and "Steps:" headers are ---

class _CatchAllFact(UnitExtractorStrategy):
    # Stand-in for DefaultNLPExtractor (needs spaCy): every sentence becomes a fact unit.
    def extract(self, index, sent):
        return [SemanticUnit(index, sent.text, "fact")]


def _workflows(text):
    extractors = [RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)"), _CatchAllFact()]
    units = build_semantic_state(segment_text(text, nlp=None), extractors)
    return [u for u in units if u.type == "workflow"], units


def test_bullet_list_under_ordinary_header_is_not_a_workflow():
    wfs, units = _workflows("Your responses should be:\n- Accurate\n- Concise\n- Friendly")
    assert wfs == []
    assert len(units) == 4  # header + three bullets, each kept as its own unit


def test_bullet_list_under_steps_header_is_a_workflow():
    wfs, _ = _workflows("Deployment steps:\n- Build\n- Push\n- Deploy")
    assert len(wfs) == 1 and len(wfs[0].steps) == 3


def test_numbered_list_under_ordinary_header_is_still_a_workflow():
    wfs, _ = _workflows("Rules:\n1. Build\n2. Push")
    assert len(wfs) == 1


def test_orphan_bullets_are_not_an_unnamed_workflow():
    wfs, units = _workflows("- one\n- two")
    assert wfs == []
    assert len(units) == 2
