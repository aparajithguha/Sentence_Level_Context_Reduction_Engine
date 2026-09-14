"""
Unit tests for scre/utils.py -- pure, stateless helpers. No spaCy pipeline
or embedder needed: every function here takes plain data in, plain data
out.
"""
from scre.utils import (
    DummySentence,
    extract_entities,
    normalize_text,
    compute_idf,
    jaccard_similarity,
    estimate_tokens,
)


def test_dummy_sentence_exposes_text():
    s = DummySentence("hello world")
    assert s.text == "hello world"


def test_extract_entities_returns_empty_set_for_object_without_ents():
    # DummySentence has no .ents -- extract_entities must degrade gracefully
    # rather than raise, since it's used in regex-only mode.
    assert extract_entities(DummySentence("PostgreSQL is a database")) == set()


class _FakeToken:
    def __init__(self, text, pos_=None):
        self.text = text
        self.pos_ = pos_


class _FakeEnt:
    def __init__(self, text):
        self.text = text


class _FakeParsedSentence:
    """Minimal stand-in for a spaCy Span exposing .ents and iteration over tokens."""
    def __init__(self, ents, tokens):
        self.ents = ents
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


def test_extract_entities_unions_ner_and_propn_tokens():
    sent = _FakeParsedSentence(
        ents=[_FakeEnt("PostgreSQL")],
        tokens=[_FakeToken("PostgreSQL"), _FakeToken("Redis", pos_="PROPN"), _FakeToken("is", pos_="AUX")],
    )
    assert extract_entities(sent) == {"postgresql", "redis"}


def test_extract_entities_strips_and_lowercases():
    sent = _FakeParsedSentence(ents=[_FakeEnt("  Kubernetes  ")], tokens=[])
    assert extract_entities(sent) == {"kubernetes"}


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world  \n\t") == "hello world"


def test_normalize_text_empty_string():
    assert normalize_text("   ") == ""


def test_compute_idf_empty_input_returns_empty_dict():
    assert compute_idf([]) == {}


def test_compute_idf_weights_rare_terms_higher_than_common_terms():
    sentences = [
        DummySentence("the database is fast"),
        DummySentence("the database is reliable"),
        DummySentence("the cache is volatile"),
    ]
    idf = compute_idf(sentences)
    # "database" appears in 2/3 docs, "volatile" in 1/3 -- rarer term must
    # carry a higher IDF weight.
    assert idf["volatile"] > idf["database"]


def test_compute_idf_skips_single_char_terms():
    idf = compute_idf([DummySentence("a b database")])
    assert "a" not in idf
    assert "b" not in idf
    assert "database" in idf


def test_jaccard_similarity_identical_strings_is_one():
    assert jaccard_similarity("the quick brown fox", "the quick brown fox") == 1.0


def test_jaccard_similarity_disjoint_strings_is_zero():
    assert jaccard_similarity("apples oranges bananas", "trucks highways engines") == 0.0


def test_jaccard_similarity_empty_string_is_zero():
    assert jaccard_similarity("", "something") == 0.0


def test_jaccard_similarity_partial_overlap():
    sim = jaccard_similarity("the quick brown fox", "the quick red fox")
    assert 0.0 < sim < 1.0


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_estimate_tokens_nonempty_is_positive_and_monotonic():
    short = estimate_tokens("hello")
    long = estimate_tokens("hello " * 50)
    assert short > 0
    assert long > short
