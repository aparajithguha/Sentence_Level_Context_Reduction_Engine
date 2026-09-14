"""
config.py
=========
Externalized scoring/graph/selection thresholds for SCRE.

Every value here was previously a literal inline in ``query_aware_reducer.py``.
Defaults match the values already in production use — this module changes
*where* they live, not what they are. Pass a custom ``ScoringConfig`` to
``SCRE(config=...)`` to tune retrieval behavior without editing engine code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Tunable thresholds for scoring, graph construction, and selection.

    Grouped by the stage of the pipeline that reads them; see
    ``scoring.score_semantic_unit``, ``graph.build_reasoning_graph``,
    and ``selection.select_semantic_units`` for exactly how each is used.
    """

    # --- Dense (embedding) similarity tiers ---
    # Below dense_mid_threshold, cosine similarity for this embedding model is
    # statistically indistinguishable from noise (verified across 4 models);
    # contributing nothing there, rather than scaling down, is intentional.
    dense_high_threshold: float = 0.65
    dense_high_multiplier: float = 12.0
    dense_mid_threshold: float = 0.45
    dense_mid_multiplier: float = 6.0

    # --- Lexical / phrase signals ---
    phrase_bonus: float = 5.0
    multi_term_boost_base: float = 1.5  # non-linear boost: base ** (n_matches - 1)

    # --- Intent / identifier / entity signals ---
    intent_bonus: float = 3.0
    workflow_intent_bonus: float = 3.0
    hard_id_boost: float = 15.0
    entity_bonus: float = 3.0
    pronoun_coreference_bonus: float = 1.0
    noun_chunk_bonus: float = 2.0

    # --- Graph connectivity ---
    connectivity_weight: float = 0.1
    connectivity_cap: float = 1.0
    # Reasoning-graph connectivity is a sparser, stronger signal than
    # knowledge-graph connectivity above (a unit either is or isn't part of
    # an actual decision/reason/constraint/goal chain, vs. knowledge-graph
    # edges which can be numerous and noisy from shared entities alone) --
    # weighted higher per edge accordingly.
    reasoning_connectivity_weight: float = 0.5
    reasoning_connectivity_cap: float = 1.5

    # --- Base weighting & preservation ---
    base_weight_multiplier: float = 2.0
    preservation_bonus: float = 2.5

    # --- Source-confidence discount ---
    # A heuristically-typed unit (DefaultNLPExtractor's keyword/POS guess)
    # competing for intent_bonus against a labeled (ground-truth) unit of
    # the same type is a guess competing with a document-asserted fact, not
    # filling a gap -- 0.0 fully suppresses its intent_bonus in that case.
    heuristic_type_discount: float = 0.0

    # --- Selection ---
    jaccard_dedup_threshold: float = 0.85
    min_tokens_default: int = 250
    # min_tokens may extend a selection past max_sentences, but never past
    # this multiple of it -- see select_semantic_units.
    selection_ceiling_multiplier: int = 2

    # --- Reasoning graph ---
    reasoning_max_dist: int = 15
    # Sentence-distance window for the lexical-causal-marker link (e.g.
    # "because"). Kept tight (1 = immediate neighbor only) -- a wider window
    # was found to link unrelated sentences on structured, densely-packed
    # documents (see tests/test_correctness.py).
    causal_window: int = 1
