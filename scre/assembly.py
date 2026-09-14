"""
assembly.py
===========
Renders the final selected/expanded units into the compressed context
string handed back to the caller.
"""
from __future__ import annotations

from .units import SemanticUnit


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

    Args:
        units: A list of ``SemanticUnit`` objects to render.  May be empty.

    Returns:
        A newline-joined string containing the rendered sentences in document
        order, or an empty string if ``units`` is empty.
    """
    # Sort units by their original position in the document to maintain narrative flow.
    ordered_units = sorted(units, key=lambda x: x.original_sentence_index)

    # Use a dictionary to ensure each original sentence appears only once,
    # and utilize the updated render_text to support Workflow truncation and JSON representation.
    unique_sentences = {}
    last_header = ""
    for u in ordered_units:
        if u.original_sentence_index not in unique_sentences:
            text_to_render = u.render_text

            # Contextual Prefixing: Inject document metadata to prevent context detachment
            if getattr(u, "context_header", "") and not u.original_sentence_text.startswith("#"):
                if u.context_header != last_header:
                    text_to_render = f"[{u.context_header}]\n{text_to_render}"
                    last_header = u.context_header

            unique_sentences[u.original_sentence_index] = text_to_render

    return "\n".join(unique_sentences.values()).strip()
