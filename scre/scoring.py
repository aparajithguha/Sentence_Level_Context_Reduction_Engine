"""
scoring.py
==========
Turns a query into a feature set (``get_query_features``) and scores every
extracted unit against it (``score_semantic_unit``) using a hybrid signal:
lexical IDF overlap, dense cosine similarity, entity/noun-chunk match,
graph connectivity, intent type-match, and hard-identifier boost.
"""
from __future__ import annotations

import re
from typing import Any

from .config import ScoringConfig
from .units import SemanticUnit, CATEGORY_WEIGHTS

ANSWER_TYPE_BOOSTS = {
    "who": {"PERSON", "ORG"},
    "where": {"GPE", "LOC", "FAC"},
    "when": {"DATE", "TIME"},
}


def get_query_features(query: str, nlp: Any) -> dict[str, Any]:
    """Parse the query into a rich feature set used for scoring.

    When spaCy is available the query is fully parsed; otherwise a
    lightweight lexical fallback is used.

    Returns a dict with the following keys:

    * ``"terms"`` (*set[str]*): Lemmatised content tokens (length ≥ 2),
      deduplicated with their surface forms.
    * ``"entities"`` (*set[str]*): Named entities recognised by spaCy
      NER (lowercased).
    * ``"noun_chunks"`` (*set[str]*): Noun phrases longer than 2
      characters (lowercased).
    * ``"wh_words"`` (*set[str]*): WH-interrogative words present in the
      query (used to select NER types for answer-type boosting).
    * ``"root_lemma"`` (*str*): Lemma of the query's syntactic root verb.
    * ``"hard_identifiers"`` (*set[str]*): Alphanumeric identifiers
      extracted via regex (e.g., ``AGENT-INT-03``, ``SOC2``,
      ``[REQ-01]``, ``OpenTelemetry``, ``v20``).
    * ``"clean_query_words"`` (*list[str]*): Lowercased, punctuation-
      stripped words used for exact-phrase matching.

    Args:
        query: The raw natural-language query string.
        nlp: A loaded spaCy ``Language`` object, or ``None`` to use the
            regex-only fallback.

    Returns:
        A feature dictionary as described above.
    """
    query_terms = []
    entities = set()
    noun_chunks = set()
    wh_words = set()
    root_lemma = ""

    if nlp:
        query_doc = nlp(query)
        for token in query_doc:
            if token.is_stop or token.is_punct or token.like_num:
                continue
            lemma = token.lemma_.lower().strip()
            text_val = token.text.lower().strip()
            if len(lemma) >= 2:
                query_terms.append(lemma)
            if len(text_val) >= 2 and text_val != lemma:
                query_terms.append(text_val)

        entities = {ent.text.lower().strip() for ent in query_doc.ents if ent.text.strip()}
        noun_chunks = {c.text.lower().strip() for c in query_doc.noun_chunks if len(c.text.strip()) > 2}
        wh_words = {t.lower_ for t in query_doc if t.lower_ in ANSWER_TYPE_BOOSTS}
        root_lemma = query_doc[:].root.lemma_.lower().strip()
    else:
        # Lexical Fallback without spaCy
        words = re.findall(r'\b\w+\b', query)
        query_terms = [w.lower() for w in words if len(w) > 2]
        wh_words = {w.lower() for w in words if w.lower() in ANSWER_TYPE_BOOSTS}

    # Extract hard identifiers (e.g., AGENT-INT-03, SOC2, [REQ-01], OpenTelemetry, v20)
    regex_pattern = r'\b[A-Z]+-[A-Z0-9\-]+\b|\b[A-Z0-9]{3,}\b|\[[A-Z]+-\d+\]|\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b|\bv\d+(?:\.\d+)*\b'
    hard_identifiers = set(re.findall(regex_pattern, query))

    clean_query_words = re.sub(r'\W+', ' ', query.lower()).split()

    return {
        "terms": set(query_terms),
        "entities": entities,
        "noun_chunks": noun_chunks,
        "wh_words": wh_words,
        "root_lemma": root_lemma,
        "hard_identifiers": hard_identifiers,
        "clean_query_words": clean_query_words,
    }


def score_semantic_unit(
    unit: SemanticUnit,
    query_features: dict[str, Any],
    idf: dict[str, float],
    dense_score: float = 0.0,
    config: ScoringConfig | None = None,
    types_with_labeled_instance: set[str] | None = None,
) -> None:
    """Compute and assign a relevance score to a single ``SemanticUnit``.

    The final score is an additive combination of the following signals:

    * **Base weight** (×2): Category importance from ``CATEGORY_WEIGHTS``.
    * **Lexical similarity**: Sum of IDF weights for matched query terms,
      with a non-linear (1.5^n) boost for multi-term matches.
    * **Phrase bonus** (+5.0): Exact query-phrase match in unit text.
    * **Dense bonus**: Scaled cosine-similarity score from the
      sentence-transformer; tiered at 0.65 / 0.45 thresholds, with
      zero contribution below 0.45 -- that range is statistically
      noise for this embedding model, not signal (verified empirically
      across multiple models), so it is excluded rather than scaled down.
    * **Intent bonus** (+3.0): Unit type matches a query keyword; further
      +3.0 if a workflow unit matches pipeline/workflow/process terms.
    * **Hard-identifier boost** (+15.0 per hit): Exact match for
      alphanumeric identifiers (e.g., ``AGENT-INT-03``, ``[REQ-01]``).
    * **Entity match** (+3.0 per entity): NER entities from the query
      found in the unit text, plus a +1.0 pronoun coreference bonus.
    * **Noun-chunk match** (+2.0 per chunk): Noun phrases from the query
      found in the unit text.
    * **Knowledge-graph connectivity**: Min-capped at 1.0, rewards units
      whose entities are mentioned elsewhere in the document too.
    * **Reasoning-graph connectivity**: Min-capped at 1.5, rewards units
      that are actually part of a decision/reason/constraint/goal chain
      with something else in the document -- a sparser, stronger signal
      than knowledge-graph connectivity, since it requires a real
      structural relationship rather than merely sharing an entity or
      restating a category word.
    * **High-priority preservation bonus** (+2.5): Applied to
      decision/constraint/goal/workflow/task/question units that have at
      least one positive signal, preventing them from being silently
      dropped.

    The score is written directly to ``unit.score``.

    Args:
        unit: The unit to score.
        query_features: Feature dict produced by ``get_query_features``.
        idf: IDF weight mapping from ``utils.compute_idf``.
        dense_score: Pre-computed cosine similarity between the query
            embedding and the unit embedding.  Pass ``0.0`` when dense
            embeddings are unavailable.
        config: Thresholds to use (see ``scre.config.ScoringConfig``).
            Defaults to ``ScoringConfig()`` when not given.
        types_with_labeled_instance: Set of ``unit.type`` values that have
            at least one ``extraction_source == "labeled"`` unit somewhere
            in the current document (computed once per ``reduce()`` call,
            not per unit). When ``unit`` is a ``"heuristic"``-typed unit
            whose type appears in this set, its intent_bonus is discounted
            via ``config.heuristic_type_discount`` -- it's a guess
            competing with a document-asserted fact of the same type, not
            filling a gap. ``None`` (default) disables the discount.
    """
    cfg = config if config is not None else ScoringConfig()
    text_lower = unit.original_sentence_text.lower()
    base_weight = CATEGORY_WEIGHTS.get(unit.type, 0.5)

    # 1. Semantic Similarity (Term Overlap weighted by IDF)
    # Category-type words (e.g. "constraint", "decision") are excluded here --
    # a unit whose *type* matches one of those words already gets credit via
    # intent_bonus below. Counting it again from raw text would double the
    # same signal, and asymmetrically so: a structured Label:-value unit's
    # rendered text never repeats its own label (RegexKeyValueExtractor keeps
    # only the value), while ordinary prose that merely *talks about* a
    # constraint/decision in the abstract often does contain that word. That
    # let generic definitional sentences ("Constraints specify what must not
    # be violated.") outscore the actual constraint they were describing,
    # purely for restating the category word intent_bonus already rewarded.
    matched_terms = [t for t in query_features["terms"] if t in text_lower and t not in CATEGORY_WEIGHTS]
    lexical_similarity = sum(idf.get(t, 1.0) for t in matched_terms)

    # Non-linear boost for matching multiple query terms (context density)
    if len(matched_terms) > 1:
        lexical_similarity *= (cfg.multi_term_boost_base ** (len(matched_terms) - 1))

    # Fix missing phrase_bonus logic
    phrase_bonus = 0.0
    query_phrase = " ".join(query_features.get("clean_query_words", []))
    if query_phrase and len(query_phrase) > 4 and query_phrase in text_lower:
        phrase_bonus = cfg.phrase_bonus

    # Dense embedding bonus to overcome lexical gaps (e.g., "unfinished" vs "pending")
    # Calibrated dense scoring so high semantic matches aren't buried by lexical noise.
    # Below dense_mid_threshold, cosine similarity between unrelated sentences is
    # statistically indistinguishable from noise for this embedding model (verified:
    # two unrelated "Decision: we chose X" sentences score ~0.2 just from shared
    # surface structure, sometimes higher than a genuinely relevant sentence with no
    # structural overlap). Rewarding that range proportionally lets noise outrank
    # real content, so it contributes nothing rather than a smoothly-scaled-down credit.
    dense_bonus = 0.0
    if dense_score > cfg.dense_high_threshold:
        dense_bonus = dense_score * cfg.dense_high_multiplier  # Scaled down to prevent total override
    elif dense_score > cfg.dense_mid_threshold:
        dense_bonus = dense_score * cfg.dense_mid_multiplier

    # 2. Intent Detection & Type Match
    intent_bonus = 0
    type_terms = set(unit.type.split('_'))
    if type_terms & query_features["terms"]:
        intent_bonus = cfg.intent_bonus  # Intent aligns with node type
        # A heuristic-typed unit's "type matches the query" credit is a
        # model's guess; when a labeled (document-asserted) unit of the
        # same type exists, the guess is competing with a fact, not filling
        # a gap the labeled unit left open -- discount it accordingly.
        if unit.extraction_source == "heuristic" and types_with_labeled_instance and unit.type in types_with_labeled_instance:
            intent_bonus *= cfg.heuristic_type_discount

    # Workflow Relevance: Boost if query asks about pipelines, workflows, or processes
    if unit.type == "workflow" and any(w in query_features["terms"] for w in ["pipeline", "workflow", "process", "steps", "stages"]):
        intent_bonus += cfg.workflow_intent_bonus

    # NEW: Hard Identifier / Keyword Boost
    hard_id_boost = 0.0
    for hid in query_features.get("hard_identifiers", set()):
        if hid in unit.original_sentence_text or hid.lower() in text_lower:
            hard_id_boost += cfg.hard_id_boost  # Massive boost for exact match of alphanumeric identifiers

    # 3. Entity & Noun Match (Broadened to include noun chunks because NER misses tech terms)
    entity_matches = [e for e in query_features["entities"] if e in text_lower]
    entity_match = sum(cfg.entity_bonus for e in entity_matches)

    noun_matches = [n for n in query_features.get("noun_chunks", set()) if n in text_lower]
    noun_match = sum(cfg.noun_chunk_bonus for n in noun_matches)

    # Generalized pronoun coreference boost if entities are queried
    if query_features["entities"] and any(p in text_lower.split() for p in ["he", "she", "it", "its", "his", "her", "they", "their"]):
        entity_match += cfg.pronoun_coreference_bonus

    # 4. Relationship Distance (Knowledge-Graph Connectivity -- shared entities)
    relationship_distance = min(unit.connectivity * cfg.connectivity_weight, cfg.connectivity_cap)

    # 5. Reasoning-Graph Connectivity -- a categorically different, sparser
    # signal: whether this unit is actually part of a decision/reason/
    # constraint/goal chain with something else in the document, as opposed
    # to merely sharing an entity or restating a category word. Generic
    # prose that only *discusses* a category in the abstract has zero
    # reasoning-graph edges and gets nothing here, however close its
    # wording sits to the query.
    reasoning_bonus = min(unit.reasoning_connectivity * cfg.reasoning_connectivity_weight, cfg.reasoning_connectivity_cap)

    unit.score = (base_weight * cfg.base_weight_multiplier) + lexical_similarity + phrase_bonus + dense_bonus + entity_match + noun_match + relationship_distance + reasoning_bonus + intent_bonus + hard_id_boost

    # Preserve critical nodes if relevant
    high_priority_types = {"decision", "constraint", "goal", "workflow", "task", "question"}
    if unit.type in high_priority_types:
        if lexical_similarity > 0 or entity_match > 0 or noun_match > 0 or dense_score > cfg.dense_mid_threshold or hard_id_boost > 0:
            preservation = cfg.preservation_bonus
            # Same rationale as the intent_bonus discount above: this bonus
            # exists to protect genuine high-priority content from being
            # dropped. A heuristic-typed unit competing against a labeled
            # unit of the same type isn't the genuine article the bonus
            # was designed to protect -- it's a guess.
            if unit.extraction_source == "heuristic" and types_with_labeled_instance and unit.type in types_with_labeled_instance:
                preservation *= cfg.heuristic_type_discount
            unit.score += preservation  # Increased preservation weight for critical logic nodes
