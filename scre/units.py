"""
units.py
========
``SemanticUnit`` and ``WorkflowUnit`` -- the atomic data model produced by
extraction (``scre.extraction``), annotated by graph-building and scoring
(``scre.graph``, ``scre.scoring``), and consumed by selection and assembly
(``scre.selection``, ``scre.assembly``). Also holds ``CATEGORY_WEIGHTS``,
the canonical set of valid unit types and their base scoring weight, since
both extraction (type canonicalization) and scoring (base weight lookup)
need it.
"""
from __future__ import annotations

import json


CATEGORY_WEIGHTS = {
    "constraint": 1.0,
    "decision": 1.0,
    "reason": 1.0,
    "implementation": 1.0,
    "goal": 0.95,
    "workflow": 0.95,
    "task": 0.90,
    "risk": 0.90,
    "outcome": 0.85,
    "alternative": 0.85,
    "comparison": 0.80,
    "question": 0.80,
    "fact": 0.70,
}


class SemanticUnit:
    """Atomic unit of extracted meaning derived from a single source sentence.

    A ``SemanticUnit`` captures the who/what/how triple of a sentence along
    with its semantic category, enabling structured scoring and graph-based
    retrieval.

    Attributes:
        original_sentence_index (int): Position of the source sentence within
            the document passed to ``reduce`` (used for ordering and graph edges).
        original_sentence_text (str): The raw text of the source sentence.
        render_text (str): The text used when assembling the final compressed
            context.  Defaults to ``original_sentence_text`` but may be
            overridden (e.g., by ``WorkflowUnit``) to a JSON representation.
        type (str): Semantic category label (e.g., ``"decision"``,
            ``"constraint"``, ``"fact"``).  Must be a key in
            ``CATEGORY_WEIGHTS`` for the base-weight lookup to apply.
        subject (str): The extracted subject phrase of the sentence.
        relation (str): The root verb or relational phrase.
        object (str): The extracted object phrase of the sentence.
        score (float): Relevance score assigned during retrieval.  Higher is
            more relevant.
        connectivity (int): Sum of knowledge-graph edge counts for the
            subject/object nodes, used as a tiebreaker during scoring.
        context_header (str): The nearest preceding Markdown heading (H1),
            injected as a prefix in the final context to prevent detachment.
        entities (set[str]): Lowercased named entities / proper nouns found
            in this unit's text, from spaCy NER and POS tagging on the
            already-parsed sentence span. Populated by
            ``extraction.build_semantic_state`` when a real spaCy parse is
            available (empty in regex-only mode). Used by
            ``graph.build_knowledge_graph`` to ground units that have no
            subject/object of their own (e.g. ``RegexKeyValueExtractor``
            output) to real entities instead of the entire sentence text.
        extraction_source (str): ``"labeled"`` if ``type`` came from an
            explicit ``Label: value`` match (ground truth), ``"heuristic"``
            (default) if it was guessed from POS/dependency parsing or
            keyword rules. Used by ``scoring.score_semantic_unit`` to avoid
            letting a guess outcompete a labeled instance of the same type
            purely on type-match credit both would otherwise get equally.
    """

    def __init__(self, original_sentence_index: int, original_sentence_text: str, unit_type: str,
                 subject: str = "", relation: str = "", obj: str = ""):
        """Initialise a SemanticUnit.

        Args:
            original_sentence_index: Zero-based index of the sentence in the
                source document.
            original_sentence_text: Raw text of the source sentence.
            unit_type: Semantic category label (e.g. ``"decision"``,
                ``"constraint"``, ``"fact"``).
            subject: Extracted subject phrase. Defaults to ``""``.
            relation: Root verb / relational phrase. Defaults to ``""``.
            obj: Extracted object phrase. Defaults to ``""``.
        """
        self.original_sentence_index = original_sentence_index
        self.original_sentence_text = original_sentence_text
        self.render_text = original_sentence_text
        self.type = unit_type.lower()
        self.subject = subject
        self.relation = relation
        self.object = obj
        # "labeled": type came from an explicit "Label: value" match
        # (RegexKeyValueExtractor) -- ground truth from the document's own
        # structure. "heuristic" (default): type was guessed from POS/
        # dependency parsing or keyword rules (DefaultNLPExtractor) -- a
        # model's inference, not a document-asserted fact. Extractors that
        # produce ground-truth typing override this after construction.
        self.extraction_source: str = "heuristic"
        self.score = 0.0
        self.connectivity = 0
        # Count of this unit's edges in the reasoning graph (decision/
        # reason/constraint/goal chains -- see graph.build_reasoning_graph),
        # as opposed to `connectivity` above (knowledge-graph entity
        # co-occurrence). Populated by graph.apply_reasoning_connectivity.
        # Zero for a unit with no structural relationship to anything else
        # in the document -- e.g. generic prose that merely discusses a
        # category in the abstract rather than being part of an actual
        # decision/constraint/reason chain.
        self.reasoning_connectivity = 0
        self.context_header = ""
        self.entities: set[str] = set()

    def __hash__(self):
        return hash((self.type, self.subject.lower(), self.relation.lower(), self.object.lower()))

    def __eq__(self, other):
        if not isinstance(other, SemanticUnit):
            return NotImplemented
        return (self.type == other.type and
                self.subject.lower() == other.subject.lower() and
                self.relation.lower() == other.relation.lower() and
                self.object.lower() == other.object.lower())


class WorkflowUnit(SemanticUnit):
    """A specialised ``SemanticUnit`` that represents an ordered procedural workflow.

    Detected when a non-list-item sentence (the *header*) is immediately
    followed by one or more numbered/bullet list items (the *steps*).  The
    ``render_text`` is serialised as a JSON object so that downstream LLMs
    receive a machine-readable representation of the procedure.

    Attributes:
        name (str): The workflow title (the header sentence, stripped of
            trailing punctuation).
        steps (list[str]): The ordered list of step strings.
    """

    def __init__(self, original_sentence_index: int, name: str, steps: list[str], original_text: str):
        """Initialise a WorkflowUnit.

        Args:
            original_sentence_index: Zero-based index of the header sentence.
            name: Workflow title derived from the header sentence.
            steps: Ordered list of procedural step strings.
            original_text: Combined raw text (header + all steps joined with
                newlines), stored as ``original_sentence_text``.
        """
        super().__init__(original_sentence_index, original_text, "workflow")
        # Structurally detected (header + list-item pattern), not guessed
        # from keywords -- ground truth, same confidence tier as a labeled
        # "Label: value" match.
        self.extraction_source = "labeled"
        self.name = name
        self.steps = steps
        self.render_text = json.dumps({
            "type": "workflow",
            "name": self.name,
            "steps": self.steps
        }, indent=2, ensure_ascii=False)

    def __hash__(self):
        return hash((super().__hash__(), tuple(self.steps)))

    def __eq__(self, other):
        if not isinstance(other, WorkflowUnit):
            return NotImplemented
        return super().__eq__(other) and self.steps == other.steps

    def get_truncated_text(self) -> str:
        return f"[Workflow '{self.name}' containing {len(self.steps)} steps omitted for brevity]"


def _find_json_workflow_spans(text: str) -> list[tuple[dict, int, int]]:
    """Scan ``text`` for ``{"type": "workflow", ...}`` JSON blocks -- how a
    preserved ``WorkflowUnit`` is rendered into a compressed context (see
    ``render_text`` above). Brace-matched rather than a single regex, since
    ``json.dumps(..., indent=2)``'s multi-line formatting would confuse a
    naive pattern.

    Shared low-level scanner behind ``extract_json_workflows`` (extraction
    only) and ``strip_json_workflows`` (extraction + removal, for callers
    that need to keep processing the surrounding text). Returns
    ``(workflow_dict, start, end)`` per match, ``end`` exclusive.
    """
    matches: list[tuple[dict, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 0
            for j in range(i, n):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and obj.get("type") == "workflow" and obj.get("steps"):
                                matches.append((obj, i, j + 1))
                        except (json.JSONDecodeError, ValueError):
                            pass
                        i = j
                        break
            else:
                break
        i += 1
    return matches


def extract_json_workflows(text: str) -> list[dict]:
    """Directly extract workflow dicts from any ``{"type": "workflow", ...}``
    JSON blocks embedded verbatim in ``text`` -- how SCRE renders a
    preserved workflow into a compressed context. Re-analyzing SCRE's own
    JSON output as if it were a fresh document (segmenting it as prose)
    shreds it via ordinary sentence segmentation and never re-detects it as
    a workflow at all, so anything checking whether a workflow survived
    reduction needs to read the JSON directly instead.
    """
    return [wf for wf, _, _ in _find_json_workflow_spans(text)]


def strip_json_workflows(text: str) -> tuple[str, list[dict]]:
    """Extract embedded workflow JSON blocks and return ``text`` with those
    spans removed, alongside the extracted dicts.

    For callers that need to keep processing the *surrounding* prose (e.g.
    ``scre_answer_engine.extractive_answer`` ranking/extracting from
    individual lines) without a workflow's JSON block shredding into
    meaningless line fragments (``'{'``, ``'"steps": ['``, ...) when the
    text is split by newline.
    """
    matches = _find_json_workflow_spans(text)
    remainder = text
    for _, start, end in reversed(matches):  # reverse: earlier spans' offsets stay valid as later ones are removed
        remainder = remainder[:start] + remainder[end:]
    return remainder, [wf for wf, _, _ in matches]
