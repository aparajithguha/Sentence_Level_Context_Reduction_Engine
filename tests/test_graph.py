"""
Unit tests for scre/graph.py -- KnowledgeGraph and reasoning-graph
construction. No spaCy pipeline or embedder needed; operates purely on
hand-built SemanticUnit data.
"""
from scre.units import SemanticUnit
from scre.config import ScoringConfig
from scre.graph import (
    KnowledgeGraph,
    build_knowledge_graph,
    apply_graph_connectivity,
    apply_reasoning_connectivity,
    build_reasoning_graph,
)


# --- KnowledgeGraph ---

def test_knowledge_graph_starts_empty():
    kg = KnowledgeGraph()
    assert len(kg) == 0
    assert kg.find_related("postgresql") == []


def test_knowledge_graph_add_and_find_related_by_subject_or_object():
    kg = KnowledgeGraph()
    kg.add("we", "chose", "postgresql", 0)
    assert len(kg) == 1
    assert kg.find_related("we") == [("we", "chose", "postgresql", 0)]
    assert kg.find_related("postgresql") == [("we", "chose", "postgresql", 0)]


def test_knowledge_graph_find_related_is_case_insensitive_and_strips():
    kg = KnowledgeGraph()
    kg.add("postgresql", "used_for", None, 0)
    assert kg.find_related("  PostgreSQL  ") == [("postgresql", "used_for", None, 0)]


def test_knowledge_graph_handles_none_object_without_indexing_it():
    kg = KnowledgeGraph()
    kg.add("postgresql", "decision", None, 0)
    assert kg.find_related("postgresql")
    # None must not be indexed as a searchable entity.
    assert kg.find_related("none") == []


# --- build_knowledge_graph ---

def test_build_knowledge_graph_uses_subject_object_triple_when_present():
    u = SemanticUnit(0, "text", "decision", subject="We", relation="Chose", obj="PostgreSQL")
    kg = build_knowledge_graph([u])
    assert kg.find_related("we")


def test_build_knowledge_graph_grounds_entity_only_units_via_entities():
    # RegexKeyValueExtractor output typically has no subject/object -- only
    # entities extracted from the sentence text ground it in the graph.
    u = SemanticUnit(0, "Constraint: Do not expose PostgreSQL credentials.", "constraint", obj="Do not expose PostgreSQL credentials.")
    u.entities = {"postgresql"}
    kg = build_knowledge_graph([u])
    related = kg.find_related("postgresql")
    assert related == [("postgresql", "constraint", None, 0)]


def test_build_knowledge_graph_skips_workflow_units():
    from scre.units import WorkflowUnit
    wf = WorkflowUnit(0, "Deploy", ["1. Build", "2. Push"], "text")
    wf.entities = {"kubernetes"}
    kg = build_knowledge_graph([wf])
    assert len(kg) == 0


# --- apply_graph_connectivity ---

def test_apply_graph_connectivity_counts_related_triples_from_other_sentences():
    u1 = SemanticUnit(0, "text", "decision", subject="we", obj="postgresql")
    u2 = SemanticUnit(1, "text", "constraint", obj="must use transactions")
    u2.entities = {"postgresql"}
    kg = build_knowledge_graph([u1, u2])
    apply_graph_connectivity(kg, [u1, u2])
    assert u1.connectivity > 0
    assert u2.connectivity > 0


def test_apply_graph_connectivity_excludes_units_own_contributed_triple():
    # Regression pin: a unit's own triple(s) must not count toward its own
    # connectivity -- otherwise merely mentioning any entity (even one
    # nobody else mentions) gives every entity-bearing unit a nonzero bump
    # that an entity-less unit doesn't get, purely from self-reference.
    lone = SemanticUnit(0, "text", "fact", obj="something")
    lone.entities = {"uniqueentity"}
    kg = build_knowledge_graph([lone])
    apply_graph_connectivity(kg, [lone])
    assert lone.connectivity == 0


# --- build_reasoning_graph ---

def test_build_reasoning_graph_links_constraint_backward_to_decision():
    # Regression pin (Bug 3): constraint/requirement looks backward for the
    # decision it constrains -- not the other direction, since a decision
    # written before its own constraint can only ever find an older,
    # unrelated constraint if it looked backward itself.
    decision = SemanticUnit(0, "Decision: use PostgreSQL.", "decision")
    constraint = SemanticUnit(1, "Constraint: must use transactions.", "constraint")
    graph = build_reasoning_graph([decision, constraint])
    assert 1 in graph[0]
    assert 0 in graph[1]


def test_build_reasoning_graph_reason_links_backward_to_decision():
    decision = SemanticUnit(0, "Decision: use PostgreSQL.", "decision")
    reason = SemanticUnit(1, "Reason: strong transactional guarantees.", "reason")
    graph = build_reasoning_graph([decision, reason])
    assert 0 in graph[1]


def test_build_reasoning_graph_respects_max_dist():
    cfg = ScoringConfig(reasoning_max_dist=2)
    decision = SemanticUnit(0, "Decision: use PostgreSQL.", "decision")
    filler = [SemanticUnit(i, f"Filler sentence {i}.", "fact") for i in range(1, 5)]
    far_constraint = SemanticUnit(5, "Constraint: must use transactions.", "constraint")
    graph = build_reasoning_graph([decision, *filler, far_constraint], config=cfg)
    # Distance (5 - 0 = 5) exceeds reasoning_max_dist=2 -- must not link.
    assert 0 not in graph[5]


def test_build_reasoning_graph_returns_entry_for_every_unit():
    units = [SemanticUnit(i, "text", "fact") for i in range(3)]
    graph = build_reasoning_graph(units)
    assert set(graph.keys()) == {0, 1, 2}


# --- build_reasoning_graph: forward-fallback (idx=51 fix) ---

def test_build_reasoning_graph_forward_fallback_links_constraint_written_before_its_decision():
    # Regression pin: a document that states the constraint *before* the
    # decision it constrains (the reverse of the Bug 3 convention) previously
    # produced zero edges for either unit -- constraint's backward search at
    # the time it's processed finds nothing (the decision hasn't been seen
    # yet), and decision never looks backward for constraint at all. This is
    # exactly the "constraint before decision" ordering that left the real
    # idx=51 document's constraint chain unconnected.
    constraint = SemanticUnit(0, "Constraint: must use transactions.", "constraint")
    decision = SemanticUnit(1, "Decision: use PostgreSQL.", "decision")
    graph = build_reasoning_graph([constraint, decision])
    assert 1 in graph[0]
    assert 0 in graph[1]


def test_build_reasoning_graph_forward_fallback_only_engages_when_backward_finds_nothing():
    # Regression pin: an in-range backward candidate must suppress the
    # forward fallback entirely, even when a forward candidate would be
    # strictly closer -- the fallback is not a "pick whichever is closer"
    # global search, only a last resort when the backward-only lookup comes
    # up completely empty. Otherwise it could silently override the Bug 3
    # backward-first result the original fix depends on.
    cfg = ScoringConfig(reasoning_max_dist=3)
    backward_decision = SemanticUnit(0, "Decision: use PostgreSQL.", "decision")  # distance 3, in range
    filler = [SemanticUnit(i, f"Filler sentence {i}.", "fact") for i in range(1, 3)]
    constraint = SemanticUnit(3, "Constraint: must use transactions.", "constraint")
    forward_decision = SemanticUnit(4, "Decision: use MongoDB.", "decision")  # distance 1, closer, but not yet seen
    graph = build_reasoning_graph([backward_decision, *filler, constraint, forward_decision], config=cfg)
    # The (farther) backward candidate is used, not the (closer) forward one.
    assert 0 in graph[3]
    assert 4 not in graph[3]


def test_build_reasoning_graph_forward_fallback_still_respects_max_dist():
    cfg = ScoringConfig(reasoning_max_dist=2)
    constraint = SemanticUnit(0, "Constraint: must use transactions.", "constraint")
    filler = [SemanticUnit(i, f"Filler sentence {i}.", "fact") for i in range(1, 5)]
    far_decision = SemanticUnit(5, "Decision: use PostgreSQL.", "decision")
    graph = build_reasoning_graph([constraint, *filler, far_decision], config=cfg)
    # No backward candidate exists at all (constraint is first), so the
    # fallback engages -- but the resulting link is still distance-checked
    # like any other, same as the existing backward-search max_dist pin.
    assert 5 not in graph[0]


# --- apply_reasoning_connectivity ---

def test_apply_reasoning_connectivity_counts_edges_from_reasoning_graph():
    decision = SemanticUnit(0, "Decision: use PostgreSQL.", "decision")
    constraint = SemanticUnit(1, "Constraint: must use transactions.", "constraint")
    reasoning_graph = build_reasoning_graph([decision, constraint])
    apply_reasoning_connectivity(reasoning_graph, [decision, constraint])
    assert decision.reasoning_connectivity == len(reasoning_graph[0])
    assert constraint.reasoning_connectivity == len(reasoning_graph[1])
    assert decision.reasoning_connectivity > 0


def test_apply_reasoning_connectivity_zero_for_unit_with_no_chain():
    # A unit that merely discusses a category in the abstract (no actual
    # structural decision/reason/constraint/goal relationship to anything)
    # must get zero -- this is the whole point of the signal being distinct
    # from knowledge-graph (entity co-occurrence) connectivity.
    lone_fact = SemanticUnit(0, "Constraints specify what must not be violated.", "fact")
    reasoning_graph = build_reasoning_graph([lone_fact])
    apply_reasoning_connectivity(reasoning_graph, [lone_fact])
    assert lone_fact.reasoning_connectivity == 0
