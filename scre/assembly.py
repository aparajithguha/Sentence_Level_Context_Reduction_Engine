"""
assembly.py
===========
Renders the final selected/expanded units into the compressed context
string handed back to the caller.
"""
from __future__ import annotations

import re

from .units import SemanticUnit

_HEADING = re.compile(r"^(#{1,6})\s+\S")
_TITLE = re.compile(r"^#\s")


def _heading_level(unit: SemanticUnit) -> int:
    """Markdown heading level (1-6) if the unit is a single heading line, else 0."""
    text = unit.original_sentence_text.strip()
    if "\n" in text:
        return 0
    m = _HEADING.match(text)
    return len(m.group(1)) if m else 0


def drop_orphan_headings(units: list[SemanticUnit], all_units: list[SemanticUnit] | None = None) -> list[SemanticUnit]:
    """Drop headings whose section has no selected body, so no empty heading is emitted.

    Headings are ordinary sentences to the selector, so adjacency expansion can pull one in without
    anything under it. A heading is kept only if some selected non-heading unit lies inside its own
    section, that is, before the next heading of the same or a higher level. ``all_units`` (every unit
    of the document, selected or not) gives the true section boundaries; without it the boundaries are
    taken from ``units`` alone, which can only keep too much, never drop a heading that has a body.

    If every unit is a heading, they are all kept rather than returning nothing. Idempotent.
    """
    ordered = sorted(units, key=lambda u: u.original_sentence_index)
    source = sorted(all_units, key=lambda u: u.original_sentence_index) if all_units is not None else ordered
    heads = [(u.original_sentence_index, _heading_level(u)) for u in source if _heading_level(u)]
    bodies = [u.original_sentence_index for u in ordered if not _heading_level(u)]

    def has_body(unit: SemanticUnit, level: int) -> bool:
        start = unit.original_sentence_index
        end = next((i for i, lvl in heads if i > start and lvl <= level), float("inf"))
        return any(start < b < end for b in bodies)

    kept = [u for u in ordered if not _heading_level(u) or has_body(u, _heading_level(u))]
    return kept or ordered


def build_compressed_context(units: list[SemanticUnit]) -> str:
    """Assemble a final compressed context string from a list of selected units.

    Behaviour:

    * Units are re-sorted by ``original_sentence_index`` to restore narrative
      order regardless of how they were scored or selected.
    * Each source sentence appears **at most once**, even if multiple
      ``SemanticUnit`` objects share the same index (e.g., overlapping
      triples extracted from the same sentence).
    * ``render_text`` is used instead of ``original_sentence_text`` so that
      ``WorkflowUnit`` objects are serialised as JSON blocks.
    * **Contextual prefixing**: When a unit's ``context_header`` changes, the
      header is injected as a ``[Section Name]`` prefix so that an LLM
      consuming the snippet can determine which document section it came from.
      The prefix goes before whatever comes first, including a ``##`` heading;
      only the ``# Title`` line itself is exempt, since it would repeat it (and it
      states the section, so the units under it are not prefixed either).
    * **No empty headings**: a heading with no selected body under it is dropped
      (see ``drop_orphan_headings``).

    Args:
        units: A list of ``SemanticUnit`` objects to render.  May be empty.

    Returns:
        A newline-joined string containing the rendered sentences in document
        order, or an empty string if ``units`` is empty.
    """
    # Sort units by their original position in the document to maintain narrative flow.
    ordered_units = sorted(drop_orphan_headings(units), key=lambda x: x.original_sentence_index)

    # Use a dictionary to ensure each original sentence appears only once,
    # and utilize the updated render_text to support Workflow truncation and JSON representation.
    unique_sentences = {}
    last_header = ""
    for u in ordered_units:
        if u.original_sentence_index not in unique_sentences:
            text_to_render = u.render_text

            # Contextual Prefixing: Inject document metadata to prevent context detachment
            if getattr(u, "context_header", ""):
                if _TITLE.match(u.original_sentence_text):
                    last_header = u.context_header      # the title line already names the section
                elif u.context_header != last_header:
                    text_to_render = f"[{u.context_header}]\n{text_to_render}"
                    last_header = u.context_header

            unique_sentences[u.original_sentence_index] = text_to_render

    return "\n".join(unique_sentences.values()).strip()
