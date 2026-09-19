"""
selection.py
============
Picks the highest-scoring non-duplicate units within a token budget
(``select_semantic_units``), then expands that selection with adjacent
sentences for coreference resolution (``expand_adjacent_context``) and
multi-hop reasoning-chain neighbors (``expand_reasoning_chains``).
"""
from __future__ import annotations

from .config import ScoringConfig
from .units import SemanticUnit
from .utils import estimate_tokens, jaccard_similarity


def select_semantic_units(
    units: list[SemanticUnit],
    max_sentences: int,
    max_tokens: int | None,
    min_tokens: int | None = None,
    config: ScoringConfig | None = None,
    reasoning_graph: dict[int, set[int]] | None = None,
) -> list[SemanticUnit]:
    """Select the highest-scoring non-duplicate units within a token budget.

    Algorithm:

    1. Filter out units with a zero or negative score.
    2. Sort remaining units by score descending. Ties are broken by
       reasoning-graph connectivity to the top-ranked unit when
       ``reasoning_graph`` is given (see below), and by original document
       order otherwise.
    3. Iterate through the ranked list, skipping duplicates (Jaccard
       similarity above ``config.jaccard_dedup_threshold`` against any
       already-selected unit, or structural equality via ``__eq__``).
    4. For each candidate, check whether adding it would exceed
       ``max_tokens``.  If it would and the unit is a ``WorkflowUnit``, try
       substituting a truncated placeholder text instead.
    5. Enforce the ``max_sentences`` cap only after the ``min_tokens`` floor
       has been reached, preventing premature truncation when the top-ranked
       units happen to be very short. The floor itself is capped at the total
       token count of every positively-scored candidate, so it can never
       force irrelevant content into the selection just to reach an arbitrary
       count on a short document -- it can only extend the selection with
       content that already scored as relevant. The floor's extension is
       additionally capped at ``config.selection_ceiling_multiplier *
       max_sentences`` units: it may pad a too-thin selection, but it can
       never override the caller's sentence budget outright and return the
       whole document.

    Reasoning-graph awareness exists because the dense-similarity noise
    floor (see ``scoring.score_semantic_unit``) and the ``min_tokens``/
    ceiling floor above both intentionally stop distinguishing among
    low-relevance candidates once no real signal separates them -- which is
    correct, but leaves ties. Breaking those ties by document order alone
    can let a reasoning-graph-*unconnected* candidate (e.g. an unrelated
    decision's own constraint) outrank a connected one (the top-ranked
    unit's own constraint) purely because it appears earlier in the
    document. This applies in two places: (a) ties within the base
    ``max_sentences`` budget prefer candidates connected to the top-ranked
    unit, and (b) once that base budget is filled, the ``min_tokens``
    floor's *extension* past it is restricted to connected candidates only
    -- it no longer pads with unrelated content just because the ceiling
    would technically allow it. An explicit ``max_sentences`` larger than
    the connected cluster's size still gets filled up to the caller's
    request, by the next-best candidate regardless of connectivity -- only
    the floor's automatic extension is gated, not the caller's own stated
    budget.

    Args:
        units: All scored ``SemanticUnit`` objects for the document.
        max_sentences: Maximum number of units to include in the selection
            (subject to the ``min_tokens`` floor override).
        max_tokens: Hard upper bound on total token count.  ``None`` disables
            token-budget enforcement.
        min_tokens: Minimum token count that must be accumulated before the
            ``max_sentences`` cap becomes effective.  Defaults to
            ``config.min_tokens_default`` when not given.  Internally capped
            at the total token count of all positively-scored candidates.
        config: Thresholds to use (see ``scre.config.ScoringConfig``).
            Defaults to ``ScoringConfig()`` when not given.
        reasoning_graph: The document's reasoning graph (from
            ``graph.build_reasoning_graph``), used only to break score ties
            in favor of candidates connected to the top-ranked unit.
            ``None`` disables this and falls back to document-order
            tie-breaking.

    Returns:
        An unordered list of selected ``SemanticUnit`` objects (caller should
        sort by ``original_sentence_index`` when order matters).
    """
    cfg = config if config is not None else ScoringConfig()
    if min_tokens is None:
        min_tokens = cfg.min_tokens_default

    positively_scored = [u for u in units if u.score > 0]

    # Tie-break key: candidates within 2 hops of the single top-ranked unit
    # in the reasoning graph sort ahead of unconnected candidates with the
    # same score. Computed once, from the graph's own top unit -- not
    # re-derived per comparison.
    connected_indices: set[int] = set()
    if reasoning_graph and positively_scored:
        top_unit = max(positively_scored, key=lambda x: x.score)
        top_idx = top_unit.original_sentence_index
        hop1 = reasoning_graph.get(top_idx, set())
        hop2 = {n for h in hop1 for n in reasoning_graph.get(h, set())}
        connected_indices = {top_idx} | hop1 | hop2

    def sort_key(u: SemanticUnit):
        is_unconnected = 0 if u.original_sentence_index in connected_indices else 1
        return (-u.score, is_unconnected)

    ranked = sorted(positively_scored, key=sort_key)
    available_tokens = sum(estimate_tokens(u.render_text) for u in ranked)
    effective_min_tokens = min(min_tokens, available_tokens)
    selected = []
    tokens = 0

    for u in ranked:
        # Once the base max_sentences budget is filled, only extend further
        # (to satisfy the min_tokens floor) with candidates connected to the
        # top-ranked unit -- extending with disconnected content just because
        # the floor wants more tokens is exactly the cross-cluster leak this
        # tie-break exists to prevent. The base budget itself stays pure
        # score ranking; this only gates the floor's *extension* past it.
        if (
            len(selected) >= max_sentences
            and reasoning_graph
            and u.original_sentence_index not in connected_indices
        ):
            continue

        is_duplicate = False
        for s in selected:
            if jaccard_similarity(u.original_sentence_text, s.original_sentence_text) > cfg.jaccard_dedup_threshold:
                is_duplicate = True
                break
            if u == s:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        u_tokens = estimate_tokens(u.render_text)

        if max_tokens and tokens + u_tokens > max_tokens:
            if getattr(u, "type", "") == "workflow" and hasattr(u, "get_truncated_text"):
                truncated_text = u.get_truncated_text()
                trunc_tokens = estimate_tokens(truncated_text)
                if tokens + trunc_tokens <= max_tokens:
                    u.render_text = truncated_text
                    selected.append(u)
                    tokens += trunc_tokens
                    if len(selected) >= max_sentences: break
                    continue
            break

        selected.append(u)
        tokens += u_tokens

        # Dynamic Thresholding: Ensure we don't cut off too early if we haven't met the minimum token floor.
        # The floor may pad a too-thin selection, but never past `selection_ceiling_multiplier` times
        # the requested sentence budget -- otherwise a low, arbitrary min_tokens floor silently
        # overrides max_sentences on short inputs.
        if len(selected) >= max_sentences:
            if tokens >= effective_min_tokens or (max_tokens and tokens >= max_tokens):
                break
            if len(selected) >= max_sentences * cfg.selection_ceiling_multiplier:
                break
    return selected


def expand_reasoning_chains(
    selected: list[SemanticUnit],
    all_units: list[SemanticUnit],
    reasoning_graph: dict[int, set[int]],
) -> list[SemanticUnit]:
    """
    Expands the selection by traversing the reasoning graph (multi-hop).
    Ensures isolated nodes pull in their complete reasoning chain.
    """
    selected_indices = {u.original_sentence_index for u in selected}
    expanded_indices = set(selected_indices)

    # 2-hop traversal to fetch the complete chain for selected nodes
    for idx in selected_indices:
        hop1 = reasoning_graph.get(idx, set())
        expanded_indices.update(hop1)
        for h1_idx in hop1:
            hop2 = reasoning_graph.get(h1_idx, set())
            expanded_indices.update(hop2)

    expanded = []
    seen: set[int] = set()
    for unit in all_units:
        if unit.original_sentence_index in expanded_indices and id(unit) not in seen:
            expanded.append(unit)
            seen.add(id(unit))

    return sorted(expanded, key=lambda x: x.original_sentence_index)


def expand_adjacent_context(
    selected: list[SemanticUnit],
    all_units: list[SemanticUnit],
    window: int,
) -> list[SemanticUnit]:
    """
    Expands the selection to include adjacent sentences based on the context window.
    Crucial for resolving coreferences like pronouns across sentence boundaries.
    """
    selected_indices = {u.original_sentence_index for u in selected}
    target_indices = set(selected_indices)

    for idx in selected_indices:
        for w in range(1, window + 1):
            target_indices.add(idx - w)
            target_indices.add(idx + w)

    expanded = list(selected)
    existing = {id(u) for u in selected}

    for u in all_units:
        if u.original_sentence_index in target_indices and id(u) not in existing:
            expanded.append(u)
            existing.add(id(u))

    return sorted(expanded, key=lambda x: x.original_sentence_index)
