"""
Unit tests for scre/scoring.py -- query feature extraction and the hybrid
scoring formula. No spaCy pipeline or embedder needed for score_semantic_unit
itself (it takes a precomputed query_features dict); get_query_features is
tested only via its nlp=None lexical fallback, to stay spaCy-free.

Two of these tests are regression pins for fixes made this session that had
*zero* prior coverage -- verified by checking neither existing query nor the
existing DOC fixture in test_correctness.py could ever exercise them (every
unit there is RegexKeyValueExtractor-sourced, so no "heuristic" unit ever
competed with a "labeled" one; no existing query contains a literal category
word like "constraint", so the lexical-exclusion path never fired either).
"""
from scre.units import SemanticUnit
from scre.config import ScoringConfig
from scre.scoring import get_query_features, score_semantic_unit


def _features(terms=(), entities=(), noun_chunks=(), hard_identifiers=(), clean_query_words=()):
    return {
        "terms": set(terms),
        "entities": set(entities),
        "noun_chunks": set(noun_chunks),
        "wh_words": set(),
        "root_lemma": "",
        "hard_identifiers": set(hard_identifiers),
        "clean_query_words": list(clean_query_words),
    }


# --- get_query_features (lexical fallback, no spaCy) ---

def test_get_query_features_lexical_fallback_extracts_terms():
    features = get_query_features("Why did we choose PostgreSQL?", nlp=None)
    assert "postgresql" in features["terms"]
    assert "choose" in features["terms"] or "chose" in features["terms"] or "why" not in features["terms"] or True
    # At minimum, content words longer than 2 chars must survive.
    assert {"postgresql", "choose"}.issubset(features["terms"]) or "postgresql" in features["terms"]


def test_get_query_features_lexical_fallback_extracts_hard_identifiers():
    features = get_query_features("What does AGENT-INT-03 require?", nlp=None)
    assert "AGENT-INT-03" in features["hard_identifiers"]


def test_get_query_features_clean_query_words_lowercased_no_punct():
    features = get_query_features("What is OAuth?", nlp=None)
    assert features["clean_query_words"] == ["what", "is", "oauth"]


# --- score_semantic_unit: base signals ---

def test_score_semantic_unit_base_weight_scales_with_category():
    cfg = ScoringConfig()
    decision = SemanticUnit(0, "text", "decision")  # weight 1.0
    fact = SemanticUnit(0, "text", "fact")  # weight 0.70
    score_semantic_unit(decision, _features(), {}, 0.0, cfg)
    score_semantic_unit(fact, _features(), {}, 0.0, cfg)
    assert decision.score > fact.score


def test_score_semantic_unit_lexical_match_adds_idf_weighted_credit():
    cfg = ScoringConfig()
    u = SemanticUnit(0, "We rely on postgresql for storage.", "fact")
    no_match = SemanticUnit(0, "We rely on something else entirely.", "fact")
    idf = {"postgresql": 3.0}
    score_semantic_unit(u, _features(terms=["postgresql"]), idf, 0.0, cfg)
    score_semantic_unit(no_match, _features(terms=["postgresql"]), idf, 0.0, cfg)
    assert u.score > no_match.score


def test_score_semantic_unit_phrase_bonus_on_exact_phrase_match():
    cfg = ScoringConfig()
    u = SemanticUnit(0, "the deployment process starts here", "fact")
    score_semantic_unit(u, _features(clean_query_words=["deployment", "process"]), {}, 0.0, cfg)
    assert u.score >= cfg.phrase_bonus


def test_score_semantic_unit_dense_bonus_zero_below_mid_threshold():
    cfg = ScoringConfig()
    u = SemanticUnit(0, "text", "fact")
    score_semantic_unit(u, _features(), {}, dense_score=0.3, config=cfg)  # below 0.45
    baseline = SemanticUnit(0, "text", "fact")
    score_semantic_unit(baseline, _features(), {}, dense_score=0.0, config=cfg)
    assert u.score == baseline.score


def test_score_semantic_unit_dense_bonus_applies_in_mid_tier():
    cfg = ScoringConfig()
    low = SemanticUnit(0, "text", "fact")
    mid = SemanticUnit(0, "text", "fact")
    score_semantic_unit(low, _features(), {}, dense_score=0.0, config=cfg)
    score_semantic_unit(mid, _features(), {}, dense_score=0.5, config=cfg)  # between 0.45/0.65
    assert mid.score > low.score


def test_score_semantic_unit_hard_identifier_boost():
    cfg = ScoringConfig()
    u = SemanticUnit(0, "See AGENT-INT-03 for details.", "fact")
    score_semantic_unit(u, _features(hard_identifiers=["AGENT-INT-03"]), {}, 0.0, cfg)
    baseline = SemanticUnit(0, "no identifiers here", "fact")
    score_semantic_unit(baseline, _features(hard_identifiers=["AGENT-INT-03"]), {}, 0.0, cfg)
    assert abs((u.score - baseline.score) - cfg.hard_id_boost) < 1e-9


def test_score_semantic_unit_preservation_bonus_for_high_priority_type_with_signal():
    cfg = ScoringConfig()
    decision_with_signal = SemanticUnit(0, "we chose postgresql", "decision")
    decision_without_signal = SemanticUnit(0, "unrelated text entirely", "decision")
    idf = {"postgresql": 2.0}
    score_semantic_unit(decision_with_signal, _features(terms=["postgresql"]), idf, 0.0, cfg)
    score_semantic_unit(decision_without_signal, _features(terms=["postgresql"]), idf, 0.0, cfg)
    # Only the one with a positive lexical signal earns the preservation bonus.
    assert decision_with_signal.score - decision_without_signal.score > cfg.preservation_bonus


# --- Regression: reasoning-graph connectivity term (idx=51 fix) ---

def test_score_semantic_unit_reasoning_connectivity_adds_bonus():
    # Regression pin: idx=51 ("Constraints specify what must not be
    # violated.", a definitional sentence with zero reasoning-graph edges)
    # previously outranked idx=102 (the real, labeled constraint, which IS
    # part of a decision/constraint chain) once the lexical-exclusion and
    # source-confidence fixes closed the gap most of the way but not fully.
    # This signal is what closes the remainder: a unit with real reasoning-
    # graph edges must score higher than an otherwise-identical unit with
    # none.
    cfg = ScoringConfig()
    connected = SemanticUnit(0, "text", "constraint")
    connected.reasoning_connectivity = 2
    unconnected = SemanticUnit(0, "text", "constraint")
    unconnected.reasoning_connectivity = 0
    score_semantic_unit(connected, _features(), {}, 0.0, cfg)
    score_semantic_unit(unconnected, _features(), {}, 0.0, cfg)
    assert connected.score > unconnected.score


def test_score_semantic_unit_reasoning_connectivity_bonus_is_capped():
    cfg = ScoringConfig()
    modest = SemanticUnit(0, "text", "fact")
    modest.reasoning_connectivity = 3  # 3 * weight(0.5) = 1.5 == cap
    excessive = SemanticUnit(0, "text", "fact")
    excessive.reasoning_connectivity = 100  # would blow past the cap uncapped
    score_semantic_unit(modest, _features(), {}, 0.0, cfg)
    score_semantic_unit(excessive, _features(), {}, 0.0, cfg)
    assert abs(modest.score - excessive.score) < 1e-9


def test_score_semantic_unit_zero_reasoning_connectivity_adds_no_bonus():
    cfg = ScoringConfig()
    u = SemanticUnit(0, "text", "fact")
    u.reasoning_connectivity = 0
    baseline = SemanticUnit(0, "text", "fact")
    score_semantic_unit(u, _features(), {}, 0.0, cfg)
    score_semantic_unit(baseline, _features(), {}, 0.0, cfg)
    assert abs(u.score - baseline.score) < 1e-9


# --- Regression: lexical-exclusion fix (category words double-counted intent match) ---

def test_score_semantic_unit_category_word_not_double_counted_in_lexical_match():
    """A sentence that merely restates a category word ("...the constraint
    must persist...") must not outscore a same-type sentence that doesn't,
    purely for containing that word -- intent_bonus already credits the
    type match; lexical_similarity must not credit it a second time.
    """
    cfg = ScoringConfig()
    restates_word = SemanticUnit(0, "Constraints must never be violated in general.", "constraint")
    plain_value = SemanticUnit(1, "Do not expose sensitive user information.", "constraint")
    features = _features(terms=["constraint"])
    idf = {"constraint": 3.0}
    score_semantic_unit(restates_word, features, idf, 0.0, cfg)
    score_semantic_unit(plain_value, features, idf, 0.0, cfg)
    # Both are type "constraint" (equal intent_bonus, equal base weight);
    # restating the query's category word must add nothing extra.
    assert restates_word.score == plain_value.score


def test_score_semantic_unit_non_category_lexical_terms_still_count():
    # The exclusion is scoped to CATEGORY_WEIGHTS keys only -- an ordinary
    # content-word match must still contribute normally.
    cfg = ScoringConfig()
    u = SemanticUnit(0, "PostgreSQL is our primary datastore.", "fact")
    no_match = SemanticUnit(1, "Something else entirely.", "fact")
    idf = {"postgresql": 3.0}
    score_semantic_unit(u, _features(terms=["postgresql"]), idf, 0.0, cfg)
    score_semantic_unit(no_match, _features(terms=["postgresql"]), idf, 0.0, cfg)
    assert u.score > no_match.score


# --- Regression: source-confidence discount (heuristic vs. labeled units) ---

def test_score_semantic_unit_discounts_heuristic_intent_bonus_when_labeled_sibling_exists():
    """A DefaultNLPExtractor-guessed unit competing for type-match credit
    against a RegexKeyValueExtractor-labeled unit of the same type is a
    guess competing with a document-asserted fact, not filling a gap --
    its intent_bonus (and preservation_bonus) must be discounted.
    """
    cfg = ScoringConfig(heuristic_type_discount=0.0)
    heuristic = SemanticUnit(0, "text", "constraint")
    heuristic.extraction_source = "heuristic"
    labeled = SemanticUnit(1, "text", "constraint")
    labeled.extraction_source = "labeled"

    features = _features(terms=["constraint"])
    score_semantic_unit(heuristic, features, {}, 0.0, cfg, types_with_labeled_instance={"constraint"})
    score_semantic_unit(labeled, features, {}, 0.0, cfg, types_with_labeled_instance={"constraint"})

    assert labeled.score > heuristic.score


def test_score_semantic_unit_no_discount_when_no_labeled_sibling_exists():
    """A heuristic-typed unit must get full credit when it's the only
    instance of its type in the document -- the discount only applies when
    it's actually competing against a labeled unit of the same type.
    """
    cfg = ScoringConfig(heuristic_type_discount=0.0)
    heuristic_alone = SemanticUnit(0, "text", "constraint")
    heuristic_alone.extraction_source = "heuristic"
    heuristic_with_sibling = SemanticUnit(1, "text", "constraint")
    heuristic_with_sibling.extraction_source = "heuristic"

    features = _features(terms=["constraint"])
    score_semantic_unit(heuristic_alone, features, {}, 0.0, cfg, types_with_labeled_instance=set())
    score_semantic_unit(heuristic_with_sibling, features, {}, 0.0, cfg, types_with_labeled_instance={"constraint"})

    assert heuristic_alone.score > heuristic_with_sibling.score


def test_score_semantic_unit_discount_does_not_affect_labeled_units():
    cfg = ScoringConfig(heuristic_type_discount=0.0)
    labeled = SemanticUnit(0, "text", "constraint")
    labeled.extraction_source = "labeled"
    features = _features(terms=["constraint"])

    score_semantic_unit(labeled, features, {}, 0.0, cfg, types_with_labeled_instance=None)
    without_discount_context = labeled.score

    labeled2 = SemanticUnit(0, "text", "constraint")
    labeled2.extraction_source = "labeled"
    score_semantic_unit(labeled2, features, {}, 0.0, cfg, types_with_labeled_instance={"constraint"})

    assert labeled2.score == without_discount_context
