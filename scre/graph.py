"""
graph.py
========
The two graphs SCRE builds per ``reduce()`` call, both held as plain
in-memory objects for the duration of that call only (never persisted):

* ``KnowledgeGraph`` -- typed ``(subject, relation, object, sentence_index)``
  triples, built by ``build_knowledge_graph`` and annotated onto units via
  ``apply_graph_connectivity``.
* The reasoning graph -- a ``{sentence_index: {connected_sentence_index}}``
  adjacency map linking decision/reason/constraint/goal/implementation
  chains, built by ``build_reasoning_graph``.
"""
from __future__ import annotations

from .config import ScoringConfig
from .units import SemanticUnit


class KnowledgeGraph:
    """A set of ``(subject, relation, object, sentence_index)`` triples,
    indexed for lookup by entity.

    Replaces an earlier design where the "graph" was just a
    ``string -> {sentence indices}`` map keyed on whatever raw subject/object
    text an extractor happened to produce -- which meant two units could
    only ever connect by having byte-identical subject/object strings, and
    units with no subject/object (most ``RegexKeyValueExtractor`` output)
    couldn't connect to anything at all. Real triples let ``find_related``
    answer "what mentions Postgres" for any unit that mentions it, however
    it was extracted.
    """

    def __init__(self):
        self.triples: list[tuple[str, str, str | None, int]] = []
        self._by_entity: dict[str, list[tuple[str, str, str | None, int]]] = {}

    def add(self, subject: str, relation: str, obj: str | None, sentence_index: int) -> None:
        """Add one triple. ``subject``/``obj`` are indexed as-is (already lowercased by the caller)."""
        triple = (subject, relation, obj, sentence_index)
        self.triples.append(triple)
        for entity in (subject, obj):
            if entity:
                self._by_entity.setdefault(entity, []).append(triple)

    def find_related(self, entity: str) -> list[tuple[str, str, str | None, int]]:
        """Return every triple whose subject or object is ``entity`` (case-insensitive)."""
        return self._by_entity.get(entity.lower().strip(), [])

    def __len__(self) -> int:
        return len(self.triples)


def build_knowledge_graph(all_units: list[SemanticUnit]) -> KnowledgeGraph:
    """Build a ``KnowledgeGraph`` of real ``(subject, relation, object)`` triples.

    Two extraction shapes are handled:

    * ``DefaultNLPExtractor`` output has a real subject/relation/object
      from dependency parsing -- used directly as one triple.
    * ``RegexKeyValueExtractor`` output (most structured "Label: value"
      documents) has no subject/object at all; the sentence's category
      (``unit.type`` -- e.g. ``"decision"``) is the only relation
      available, so each entity found in the sentence (``unit.entities``)
      becomes its own ``(entity, unit.type, None)`` triple. This is what
      lets, e.g., a "Decision: chose PostgreSQL" sentence and a
      "Constraint: must use transactions" sentence connect through
      shared entities even though neither has a structured object.
    """
    graph = KnowledgeGraph()
    for unit in all_units:
        if unit.type == "workflow":
            continue
        if unit.subject and unit.object:
            graph.add(unit.subject.lower(), unit.relation.lower() or "related_to", unit.object.lower(), unit.original_sentence_index)
        for entity in unit.entities:
            graph.add(entity, unit.type, None, unit.original_sentence_index)
    return graph


def apply_graph_connectivity(graph: KnowledgeGraph, all_units: list[SemanticUnit]) -> None:
    """Annotate each unit with a graph-connectivity score.

    For each unit, the connectivity score is the number of knowledge-graph
    triples from OTHER sentences involving any entity this unit is about
    (its subject, object, or extracted entities). A higher score means
    those entities are mentioned elsewhere in the document too, which
    serves as a proxy for importance. A unit's own contributed triple(s)
    are excluded -- otherwise every unit that merely mentions any entity
    at all (even one nobody else mentions) gets a nonzero bump purely
    from referencing itself, which is presence, not connectivity, and
    wrongly outranks entity-less units (e.g. a constraint with no proper
    nouns) that may be just as relevant.

    The value is written directly to ``unit.connectivity`` and later used
    as a small additive bonus in ``scoring.score_semantic_unit``.

    Args:
        graph: The ``KnowledgeGraph`` returned by ``build_knowledge_graph``.
        all_units: All ``SemanticUnit`` objects for the document.
    """
    for unit in all_units:
        nodes = {unit.subject.lower(), unit.object.lower()} | unit.entities
        connectivity = 0
        for node in nodes:
            if node:
                connectivity += sum(
                    1 for triple in graph.find_related(node)
                    if triple[3] != unit.original_sentence_index
                )
        unit.connectivity = connectivity


def apply_reasoning_connectivity(reasoning_graph: dict[int, set[int]], all_units: list[SemanticUnit]) -> None:
    """Annotate each unit with its reasoning-graph edge count.

    This is a distinct signal from ``apply_graph_connectivity``'s knowledge-
    graph connectivity: that one reflects shared *entities* (topical
    co-occurrence, often noisy -- many sentences can share an entity without
    being related to each other). This one reflects an actual structural
    decision/reason/constraint/goal chain edge (see
    ``build_reasoning_graph``), a categorically stronger and sparser signal
    -- a unit either is or isn't part of a real reasoning chain with
    something else in the document. A unit with zero reasoning-graph edges
    (e.g. generic prose that discusses a category in the abstract, like
    "Constraints specify what must not be violated.", rather than being an
    actual constraint in a decision record) gets nothing here, regardless
    of how topically similar its wording is to a query.

    The value is written directly to ``unit.reasoning_connectivity`` and
    used as a small additive bonus in ``scoring.score_semantic_unit``.

    Args:
        reasoning_graph: The graph returned by ``build_reasoning_graph``.
        all_units: All ``SemanticUnit`` objects for the document.
    """
    for unit in all_units:
        unit.reasoning_connectivity = len(reasoning_graph.get(unit.original_sentence_index, set()))


def build_reasoning_graph(all_units: list[SemanticUnit], config: ScoringConfig | None = None) -> dict[int, set[int]]:
    """
    Constructs a graph of reasoning chains (e.g., Decision -> Reason -> Implementation).
    Generates edges for multi-hop retrieval.

    Each structural rule below looks *backward* for its natural
    predecessor: a reason follows its decision, a constraint/requirement
    follows the decision it constrains, an implementation/task follows
    the decision or reason it satisfies. Backward-looking is only
    correct when the predecessor is actually written first -- a decision
    looking backward for "the last constraint" can only ever find an
    older, unrelated one, since its own constraint (if any) hasn't been
    written yet at that point. That directional mismatch previously let
    one decision's constraint attach to a *different*, unrelated later
    decision purely by proximity.

    ``last_seen`` retains every recent unit per type (within
    ``MAX_DIST`` sentences) so history from an earlier same-type unit
    isn't lost when a newer one appears, but each lookup links to only
    the *nearest* candidate -- linking to every retained candidate would
    reintroduce the same cross-contamination in the mirrored direction
    (e.g. a later decision's constraint reattaching to an earlier,
    unrelated decision).

    A backward search can come up completely empty even when the true
    predecessor exists just a sentence away, if the document happens to
    write it in the *other* order (e.g. constraints listed before the
    decision they constrain, rather than after). That's not a wrong link
    the way the original cross-contamination bug was -- it's a missed one.
    ``link_nearest`` falls back to a nearest-in-either-direction search,
    but only when the backward search finds nothing at all, so it never
    competes with or overrides the backward-first result the original fix
    depends on for documents that do follow that convention.

    Args:
        all_units: All ``SemanticUnit`` objects for the document, in
            document order.
        config: Thresholds to use (see ``scre.config.ScoringConfig``).
            Defaults to ``ScoringConfig()`` when not given.
    """
    cfg = config if config is not None else ScoringConfig()
    MAX_DIST = cfg.reasoning_max_dist
    reasoning_graph: dict[int, set[int]] = {u.original_sentence_index: set() for u in all_units}
    last_seen: dict[str, list[SemanticUnit]] = {}

    # Full, order-independent index for the forward fallback below -- built
    # once, up front, since that fallback needs visibility into units the
    # backward-only main loop (which walks forward, accumulating last_seen
    # as it goes) hasn't reached yet.
    by_type: dict[str, list[SemanticUnit]] = {}
    for u in all_units:
        by_type.setdefault(u.type, []).append(u)

    causal_markers = [
        "because", "due to", "causes", "results in", "enables",
        "requires", "depends on", "implements", "satisfies",
        "violates", "fulfills", "supports", "selected because", "chosen because"
    ]

    def link_if_close(u1, u2, max_dist=MAX_DIST):
        if u1 and u2 and abs(u1.original_sentence_index - u2.original_sentence_index) <= max_dist:
            reasoning_graph[u1.original_sentence_index].add(u2.original_sentence_index)
            reasoning_graph[u2.original_sentence_index].add(u1.original_sentence_index)

    def nearest_any_direction(unit, unit_type):
        candidates = [c for c in by_type.get(unit_type, []) if c is not unit]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(unit.original_sentence_index - c.original_sentence_index))

    def link_nearest(unit, unit_type):
        candidates = last_seen.get(unit_type, [])
        nearest = (
            min(candidates, key=lambda c: abs(unit.original_sentence_index - c.original_sentence_index))
            if candidates else nearest_any_direction(unit, unit_type)
        )
        link_if_close(unit, nearest)

    for i, unit in enumerate(all_units):
        # Prune entries that have fallen outside the linking window so
        # `last_seen` stays bounded rather than growing for the whole doc.
        for t in list(last_seen.keys()):
            last_seen[t] = [
                u for u in last_seen[t]
                if abs(unit.original_sentence_index - u.original_sentence_index) <= MAX_DIST
            ]

        # Explicit structural relationships -- each looks backward for its
        # natural predecessor (see docstring on lookup direction).
        if unit.type == "reason":
            link_nearest(unit, "decision")
        elif unit.type == "decision":
            link_nearest(unit, "goal")
        elif unit.type in ["constraint", "risk", "alternative"]:
            link_nearest(unit, "decision")
        elif unit.type in ["implementation", "task"]:
            link_nearest(unit, "decision")
            link_nearest(unit, "reason")
            link_nearest(unit, "goal")

        # Lexical causal markers
        text_lower = unit.original_sentence_text.lower()

        if any(marker in text_lower for marker in causal_markers):
            # O(N^2) refactored to O(N) using sliding window boundaries
            start_idx = max(0, i - 5)
            end_idx = min(len(all_units), i + 6)
            for j in range(start_idx, end_idx):
                if i != j:
                    other = all_units[j]
                    if abs(other.original_sentence_index - unit.original_sentence_index) <= cfg.causal_window:
                        reasoning_graph[unit.original_sentence_index].add(other.original_sentence_index)
                        reasoning_graph[other.original_sentence_index].add(unit.original_sentence_index)

        last_seen.setdefault(unit.type, []).append(unit)

    return reasoning_graph
