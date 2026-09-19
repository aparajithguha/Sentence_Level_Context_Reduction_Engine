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


@dataclass(frozen=True)
class PromptConfig:
    """Thresholds for ``scre.prompt_reducer.PromptReducer`` (system-prompt mode).

    Everything not covered by a rule below is kept: the prompt reducer only
    ever drops trimmed examples, near-duplicate instructions, horizontal
    rules and the headings/labels/tag pairs left empty by those drops.
    """

    # Keep at most this many examples per group of sibling examples.
    example_keep: int = 2
    # Word-set Jaccard similarity at/above which a later instruction is a
    # duplicate of an earlier one (only blocks at least ``duplicate_min_chars`` long).
    duplicate_threshold: float = 0.9
    duplicate_min_chars: int = 40
    # Strip trailing spaces and collapse 3+ newlines (never inside code).
    normalize_whitespace: bool = True
    drop_horizontal_rules: bool = True

    # Task-based module dropping (only when a task is given to ``reduce``).
    # A section with two or more child sections is split into them once it
    # exceeds ``module_split_chars``; modules under ``module_min_chars`` are
    # always kept; a task term found in at most this share of the sections
    # counts as matching the section it appears in.
    module_split_chars: int = 1500
    module_min_chars: int = 400
    module_max_df_ratio: float = 0.5

    # ``reduce(selector=...)``: how much of each section's text the selector sees, and what to do if the
    # selector raises. "keep" -> redundancy-only reduction (safest); "match" -> the built-in task matcher.
    selector_text_chars: int = 3000
    on_selector_error: str = "keep"
