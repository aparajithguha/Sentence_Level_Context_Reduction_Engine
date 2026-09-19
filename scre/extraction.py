"""
extraction.py
=============
Turns raw document text into a flat, ordered list of ``SemanticUnit``
objects: sentence segmentation (``segment_text``), then per-sentence
classification via a pluggable Chain-of-Responsibility extractor chain
(``UnitExtractorStrategy`` and its implementations), assembled into
document-ordered units with workflow-grouping and heading tracking
(``build_semantic_state``).
"""
from __future__ import annotations

import re
from typing import Any

from .units import SemanticUnit, WorkflowUnit, CATEGORY_WEIGHTS
from .utils import DummySentence, extract_entities, normalize_text

_PROCEDURE_HEADER_RE = re.compile(r"\b(steps?|workflow|procedure|process|stages?|pipeline|phases?|sequence)\b", re.IGNORECASE)


def segment_text(text: str, nlp: Any) -> list:
    """Split raw document text into sentence-like objects.

    Pre-processes multi-line ``Label:\\n Value`` pairs (merges them so
    sentence tokenisers don't see them as unrelated lines), then segments
    each paragraph into sentences via spaCy (or a regex fallback when
    ``nlp`` is ``None``). List items, structured ``Label: value`` lines, and
    headers are kept intact as single sentences so the extractor chain never
    sees two distinct categories (e.g. Decision + Reason) fused into one.

    Args:
        text: Raw document/prompt text.
        nlp: A loaded spaCy ``Language`` object, or ``None`` to use the
            regex-only fallback.

    Returns:
        An ordered list of sentence-like objects (spaCy spans or
        ``DummySentence`` instances), each exposing a ``.text`` attribute.
    """
    # Pre-process text to handle multi-line "Label: Value" structures,
    # which are common in structured documents but can be broken by sentence tokenizers.
    lines = text.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        original_line = lines[i]
        line_stripped = original_line.strip()
        # Heuristic: Check if the line looks like a label (e.g., "Future Action:")
        # It ends with a colon, and only contains one colon.
        if line_stripped.endswith(':') and line_stripped.count(':') == 1:
            # If there is a non-empty next line, merge them.
            if i + 1 < len(lines) and lines[i+1].strip():
                next_line_stripped = lines[i+1].strip()
                # Do not merge if the next line is a list item
                if not re.match(r'^\d+\.\s|^[\*\-]\s', next_line_stripped):
                    leading_whitespace = original_line[:len(original_line) - len(original_line.lstrip())]
                    merged_line = leading_whitespace + line_stripped + ' ' + next_line_stripped
                    new_lines.append(merged_line)
                    i += 2  # Skip the next line as it has been merged
                    continue
        new_lines.append(original_line)
        i += 1
    processed_text = '\n'.join(new_lines)

    # A line matching this shape is a structured "Label: value" statement
    # (SDLC-tagged or Markdown key-value). Such lines must never be handed
    # to spaCy merged with a neighboring line, or the extractor chain
    # collapses two distinct categories (e.g. Decision + Reason) into one.
    is_list_item = lambda t: bool(re.match(r'^\d+\.\s|^[\*\-]\s', t))
    is_structured_line = lambda t: bool(re.match(r'^(?:#+\s*)?(?:-\s*)?(?:\[[A-Z]+-\d+\]\s*)?[\w\s-]+:\s*\S', t))

    # 1. Sentence Segmentation
    sentences = []
    for paragraph in processed_text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_lines = paragraph.split('\n')
        # A list-item or structured-line match only justifies switching this
        # whole paragraph into line-by-line atomic mode when there's more
        # than one line to potentially merge across -- the failure this
        # guards against (spaCy not splitting at a line break between two
        # Label: lines, or between a header and its list items) can't happen
        # in a single-line paragraph, and forcing it there would wrongly
        # keep an ordinary multi-sentence line (e.g. "Note: do X. Then do
        # Y.") from being segmented. This previously applied only to
        # is_structured_line, not is_list_item: a document with no blank
        # lines at all (so the whole text is one `para_lines` entry) whose
        # first character happened to be "-", "*", or "1." was treated as a
        # single list item and never split at all, however long it was --
        # e.g. a system prompt pasted as one unbroken block starting with a
        # bullet collapsed into one unsplittable "sentence".
        has_forced_lines = len(para_lines) > 1 and (
            any(is_list_item(l.strip()) for l in para_lines)
            or any(is_structured_line(l.strip()) for l in para_lines)
        )

        if has_forced_lines:
            for line in para_lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # Force list items, structured Label:-value lines, and
                # potential headers to remain intact as single sentences.
                if is_list_item(line_str) or is_structured_line(line_str) or line_str.endswith(':'):
                    if nlp:
                        doc = nlp(line_str)
                        sentences.append(doc[:])
                    else:
                        sentences.append(DummySentence(line_str))
                else:
                    if nlp:
                        doc = nlp(line_str)
                        for sent in doc.sents:
                            if normalize_text(sent.text):
                                sentences.append(sent)
                    else:
                        for sent_text in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line_str):
                            if normalize_text(sent_text):
                                sentences.append(DummySentence(sent_text))
        else:
            # Normal paragraph, let spaCy do its sentence segmentation
            if nlp:
                doc = nlp(paragraph)
                for sent in doc.sents:
                    if normalize_text(sent.text):
                        sentences.append(sent)
            else:
                for sent_text in re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph):
                    if normalize_text(sent_text):
                        sentences.append(DummySentence(sent_text))

    return sentences


class UnitExtractorStrategy:
    """Base protocol for semantic text extraction strategies."""
    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        raise NotImplementedError


class RegexKeyValueExtractor(UnitExtractorStrategy):
    """Extracts semantic units using a configurable regex pattern.

    Supports two structural formats:

    * **SDLC format** – ``[DEC-01] Title: Description`` (3 capture groups):
      the tag prefix is looked up in ``type_map`` to determine the unit type.
    * **Markdown Key-Value format** – ``Key: Value`` (2 capture groups):
      the key is normalised and canonicalised to a ``CATEGORY_WEIGHTS`` type.

    If the pattern does not match the sentence, the extractor returns an
    empty list and the next strategy in the chain is tried.
    """

    def __init__(self, pattern: str, type_map: dict[str, str] = None, default_type: str = "fact"):
        """Initialise the extractor.

        Args:
            pattern: A regex pattern string with 2 or 3 capture groups.
            type_map: Optional mapping from tag prefix (e.g. ``"DEC"``) to
                unit type label.  Only used when the pattern has 3 groups.
            default_type: Fallback unit type when the tag is not in
                ``type_map``.  Defaults to ``"fact"``.
        """
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.type_map = type_map or {}
        self.default_type = default_type

    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        text = sent.text.strip()
        match = self.pattern.match(text)
        if match:
            groups = match.groups()
            # Handles SDLC formats e.g. [DEC-01] Title: Description
            if len(groups) == 3:
                doc_type, title, value = groups
                unit_type = self.type_map.get(doc_type.upper(), self.default_type)
                unit = SemanticUnit(index, text, unit_type, subject=title.strip(), obj=value.strip())
                unit.extraction_source = "labeled"
                return [unit]
            # Handles Markdown Key: Value formats
            elif len(groups) == 2:
                label, value = groups
                unit_type = normalize_text(label).replace(" ", "_").lower()

                # Canonicalize memory types
                if unit_type in ["requirement", "rule", "non-goal", "non-goals", "non_goal", "non_goals"]: unit_type = "constraint"
                elif unit_type in ["future_action", "action"]: unit_type = "task"
                elif unit_type in ["open_question"]: unit_type = "question"
                elif unit_type in ["risk", "risks", "drawback", "drawbacks"]: unit_type = "risk"
                elif unit_type in ["alternative", "alternatives", "alternatives_considered", "alternative_considered"]: unit_type = "alternative"
                elif unit_type not in CATEGORY_WEIGHTS:
                    unit_type = "fact"

                if value:
                    unit = SemanticUnit(index, text, unit_type, obj=value)
                    unit.extraction_source = "labeled"
                    return [unit]
        return []


class DefaultNLPExtractor(UnitExtractorStrategy):
    """Fallback semantic extraction using SpaCy POS tagging and Dependency Parsing."""
    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        units = []
        text = sent.text.strip()
        text_lower = text.lower()

        def get_phrase(token):
            if not token: return ""
            return " ".join([t.text for t in token.subtree]).strip()

        subj_token = next((t for t in sent if "subj" in t.dep_ and t.head == sent.root), None)
        main_subject = get_phrase(subj_token)
        obj_token = next((t for t in sent if t.dep_ in {"dobj", "pobj", "attr", "oprd", "acomp", "xcomp"} and t.head == sent.root), None)
        main_object = get_phrase(obj_token)
        relation = sent.root.text

        # A prohibition ("do not alter his face", "don't use cloud APIs")
        # is a constraint regardless of which negating word carries it --
        # checking the dependency parse for a `neg` edge (spaCy tags
        # "not"/"n't" this way on whatever verb they attach to) catches
        # that meaning structurally, instead of a fixed literal-word list
        # that only matches "must"/"never"/etc. and misses "do not X",
        # "don't X", "isn't X", "won't X", and any future rephrasing of
        # the same prohibition.
        has_negation = any(t.dep_ == "neg" for t in sent)

        unit_type = "fact"
        if any(w in text_lower for w in ["chose", "decided", "selected", "opted", "resolved"]): unit_type = "decision"
        elif any(w in text_lower for w in ["goal", "protect", "preserve", "mission", "purpose", "aim"]): unit_type = "goal"
        # "because" states a justification -- the same role a labeled
        # "Reason:" line plays -- so it maps to "reason", not "outcome".
        # Previously both shared the "outcome" bucket, so a heuristically
        # detected "X because Y" scored lower (0.85) than a document that
        # happened to spell the same justification as "Reason: Y" (1.0),
        # purely because of which words carried it. "resulted"/"caused"/
        # "benefits"/"consequence" describe an effect, not a justification
        # -- a genuinely different, still-outcome-shaped concept -- so
        # they stay put.
        elif "because" in text_lower: unit_type = "reason"
        elif any(w in text_lower for w in ["resulted", "caused", "benefits", "outcome", "consequence"]): unit_type = "outcome"
        elif any(w in text_lower for w in ["risk", "downside", "trade-off", "tradeoff"]): unit_type = "risk"
        elif any(w in text_lower for w in ["alternative", "rejected in favor of"]): unit_type = "alternative"
        elif has_negation or any(w in text_lower for w in ["must", "cannot", "required", "limited", "never", "avoid", "prohibited", "forbidden", "restricted", "mandatory"]): unit_type = "constraint"
        elif any(w in text_lower for w in ["compare", "vs", "versus", "better than", "faster", "lighter"]): unit_type = "comparison"
        elif any(w in text_lower for w in ["task", "implement", "benchmark", "create", "generate"]): unit_type = "task"
        elif text_lower.strip().endswith("?") or "open question" in text_lower: unit_type = "question"

        if main_subject or relation or main_object:
            units.append(SemanticUnit(index, sent.text, unit_type, main_subject, relation, main_object))
        else:
            units.append(SemanticUnit(index, sent.text, "fact", "", "", sent.text))

        for token in sent:
            if token.dep_ == "advcl" and token.tag_ == "TO":
                p_obj_token = next((c for c in token.children if c.dep_ == "dobj"), None)
                p_obj = get_phrase(p_obj_token)
                if token.text and p_obj:
                    units.append(SemanticUnit(index, sent.text, "goal", main_subject, token.text, p_obj))
            elif token.dep_ == "advcl" and token.text.lower() == "because":
                # A "because" clause states a justification, the same role
                # as a labeled "Reason:" line -- see the primary
                # classification above for why this isn't "outcome".
                c_subj = get_phrase(next((c for c in token.head.children if "subj" in c.dep_), None))
                if c_subj:
                    units.append(SemanticUnit(index, sent.text, "reason", c_subj, token.head.text, get_phrase(token.head)))

        return units


def _extract_units_from_sentence(index: int, sent: Any, extractors: list[UnitExtractorStrategy]) -> list[SemanticUnit]:
    """Run the extractor chain on a single sentence and return the first successful result.

    Strategies are tried in the order they were registered.  The first
    strategy that returns a non-empty list wins; remaining strategies are
    skipped.  This mirrors the Chain-of-Responsibility pattern.

    Args:
        index: Zero-based sentence index within the source document.
        sent: A sentence-like object with a ``.text`` attribute.
        extractors: The ordered chain of extractor strategies to try.

    Returns:
        A list of ``SemanticUnit`` objects from the winning extractor, or
        an empty list if no extractor matched.
    """
    for extractor in extractors:
        extracted = extractor.extract(index, sent)
        if extracted:
            return extracted
    return []


def build_semantic_state(sentences: list, extractors: list[UnitExtractorStrategy]) -> list[SemanticUnit]:
    """Convert a flat list of sentences into a structured list of ``SemanticUnit`` objects.

    Responsibilities:

    * Tracks Markdown H1 headings and stamps each unit with a
      ``context_header`` so isolated snippets can be re-anchored to their
      document section during rendering.
    * Detects *named workflows*: a non-list header sentence followed
      immediately by one or more list items is collapsed into a single
      ``WorkflowUnit``.
    * Detects *anonymous workflows*: list items that appear without a
      preceding header are grouped under an ``"Unnamed Workflow"`` unit.
    * All remaining sentences are routed through the extractor chain.

    Args:
        sentences: Ordered list of sentence-like objects with a ``.text``
            attribute (spaCy spans or ``DummySentence`` instances).
        extractors: The ordered chain of extractor strategies to try per
            sentence.

    Returns:
        A flat list of ``SemanticUnit`` (including ``WorkflowUnit``)
        objects in document order.
    """
    all_extracted_units = []
    current_h1 = ""

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        text = sent.text.strip()

        # Hierarchical Context Tracking: Preserve the top-level document subject
        if text.startswith("# "):
            clean_header = text.strip("# \t")
            if i > 0 and sentences[i-1].text.strip().startswith("# "):
                current_h1 += f" | {clean_header}"
            else:
                current_h1 = clean_header

        is_list_item = lambda t: bool(re.match(r'^\d+\.\s|^[\*\-]\s', t)) and not bool(re.match(r'^(?:-\s*)?\[[A-Z]+-\d+\]', t))
        is_numbered = lambda t: bool(re.match(r'^\d+\.\s', t))
        is_hr = lambda t: bool(re.match(r'^[-\*=_\s]{3,}$', t))

        # Workflow Detection: Header followed by ordered procedural steps.
        # A numbered list is a procedure; a bullet list is only one when its
        # header says so ("Steps:", "Workflow:", ...). Any other bullet list
        # is an ordinary set of statements -- wrapping it as a JSON workflow
        # misrepresents it as ordered steps and inflates short inputs.
        if i + 1 < len(sentences) and not is_list_item(text) and not is_hr(text) and is_list_item(sentences[i+1].text.strip()):
            header_text = text.rstrip(':').strip()
            steps = []
            j = i + 1
            while j < len(sentences) and is_list_item(sentences[j].text.strip()):
                steps.append(sentences[j].text.strip())
                j += 1

            if is_numbered(steps[0]) or _PROCEDURE_HEADER_RE.search(header_text):
                workflow_text = text + "\n" + "\n".join(steps)
                wf_unit = WorkflowUnit(i, header_text, steps, workflow_text)
                wf_unit.context_header = current_h1
                all_extracted_units.append(wf_unit)
                i = j
                continue

        # Fallback Workflow Detection: numbered list items without a valid
        # preceding header. Bullets without a procedure header fall through
        # to ordinary per-sentence extraction below.
        if is_numbered(text):
            header_text = "Unnamed Workflow"
            steps = [text]
            j = i + 1
            while j < len(sentences) and is_list_item(sentences[j].text.strip()):
                steps.append(sentences[j].text.strip())
                j += 1

            workflow_text = "\n".join(steps)
            wf_unit = WorkflowUnit(i, header_text, steps, workflow_text)
            wf_unit.context_header = current_h1
            all_extracted_units.append(wf_unit)
            i = j
            continue

        extracted = _extract_units_from_sentence(i, sent, extractors)
        sent_entities = extract_entities(sent)
        for unit in extracted:
            unit.context_header = current_h1
            unit.entities = sent_entities
        all_extracted_units.extend(extracted)
        i += 1

    return all_extracted_units
