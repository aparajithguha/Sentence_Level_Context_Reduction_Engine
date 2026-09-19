"""
reducer_adapter.py
===================
A consumer-side wrapper around the ``scre`` library. Adds two things the
public ``SCRE.reduce()`` API deliberately does not return:

1. Per-sentence scoring/kept-or-dropped detail, for a "how it reduced"
   panel.
2. An LLM-judged comparison of what a model takes away from the original
   content versus the reduced content.

Built entirely from ``scre``'s own public module surface -- ``SCRE`` for
its loaded ``nlp``/``embedder``/``config``/``extractors``, plus the same
``scre.extraction`` / ``scre.graph`` / ``scre.scoring`` / ``scre.selection``
/ ``scre.assembly`` functions ``SCRE.reduce()`` itself composes internally.
Nothing here reaches into a private (``_``-prefixed) method or edits a
file under ``scre/`` -- this module could just as well live outside this
repository, importing ``scre`` as an installed dependency.
"""
from __future__ import annotations

from typing import Any

from scre.query_aware_reducer import SCRE
from scre.extraction import segment_text, build_semantic_state
from scre.graph import (
    build_knowledge_graph,
    apply_graph_connectivity,
    apply_reasoning_connectivity,
    build_reasoning_graph,
)
from scre.scoring import get_query_features, score_semantic_unit
from scre.selection import select_semantic_units, expand_adjacent_context, expand_reasoning_chains
from scre.assembly import build_compressed_context, drop_orphan_headings
from scre.utils import DummySentence, compute_idf, estimate_tokens
from scre.scre_answer_engine import answer_with_ollama, judge_meaning_retention

# SCRE's reduction is query-aware; this UI has no query field, so every
# call scores against this fixed, generic query instead. It doesn't ask
# about any specific document content, so lexical/entity/dense-similarity
# signals mostly stay at zero and ranking falls back to what's left:
# category base-weight (decision/constraint/reason score highest) and
# reasoning/knowledge-graph connectivity -- i.e. "keep the structurally
# important, connected content," which is the closest thing to a sensible
# default for a query-less "reduce this" request.
DEFAULT_QUERY = "What are the key decisions, constraints, reasons, goals, and workflow steps?"

# Used only for the "what the model takes away" comparison -- a fixed
# question so the same prompt is asked of both the original and the
# reduced content, making the two answers directly comparable.
UNDERSTANDING_QUESTION = "In one or two sentences, what is the key takeaway from this content?"

MODEL_NAMES = {
    "qwen3": "qwen3:4b-instruct-2507-q4_K_M",
    "gemma4": "gemma4:e2b",
}

# A document is confidently paragraph-style ("unstructured_prompt") only once
# its sentence count clearly exceeds its raw line count -- multiple
# sentences packed into the same line/paragraph, the shape of flowing
# prose rather than one bullet/label per line.
PARAGRAPH_DENSITY_THRESHOLD = 1.5
# Below this many sentences, density alone is too noisy to trust -- e.g.
# a single two-sentence greeting has density 2.0 (over threshold) despite
# being far too short to confidently call "paragraph-style" anything.
# Short, ambiguous input falls back to the "unclear -> system_prompt"
# default instead.
MIN_SENTENCES_FOR_UNSTRUCTURED_PROMPT = 4


def classify_prompt_type(text: str, units: list[dict[str, Any]], original_sentences: int) -> str:
    """Classify raw input as a structured ``"system_prompt"`` or a paragraph-style ``"unstructured_prompt"``.

    This is a policy decision about how to *use* the library, not a
    change to how it extracts or scores -- it decides which shape the
    input looks like, not whether that content is a decision or a fact.
    It lives here, not in ``scre/``, for the same reason the rest of this
    module does.

    A unit only counts as genuine structure when it is both
    ``extraction_source == "labeled"`` (a ``Label: value`` line or a
    detected numbered/bulleted workflow -- see ``scre.extraction``) *and*
    its label normalized to a real category (``type != "fact"``).
    Without the second check, an ordinary sentence that merely contains a
    colon for rhetorical effect (e.g. "Show realistic details: narrow
    aisle, ...") would count as "structured" too, since
    ``RegexKeyValueExtractor``'s pattern matches any ``clause: rest``
    shape -- it only becomes a real signal once the label itself resolves
    to a recognized category (``decision``, ``constraint``, ``reason``,
    ``goal``, ``workflow``, ``task``, ...), which prose colons don't.

    Falls back to ``"system_prompt"`` whenever the signal is weak or
    absent -- the safer default per the same reasoning as SCRE's own
    "unclear content stays" bias: a paragraph-style document wrongly
    treated as structured only risks a smaller-than-ideal reduction, while
    the reverse risks dropping content a generative prompt actually needs.
    """
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return "system_prompt"

    structured_unit_count = sum(
        1 for u in units
        if _unit_field(u, "extraction_source") == "labeled" and _unit_field(u, "type") != "fact"
    )
    if structured_unit_count > 0:
        return "system_prompt"

    density = original_sentences / len(non_empty_lines)
    if original_sentences >= MIN_SENTENCES_FOR_UNSTRUCTURED_PROMPT and density >= PARAGRAPH_DENSITY_THRESHOLD:
        return "unstructured_prompt"
    return "system_prompt"


def _unit_field(unit: Any, field: str) -> Any:
    """Read ``field`` from either a ``SemanticUnit`` or the plain dict shape ``units_out`` uses."""
    return unit[field] if isinstance(unit, dict) else getattr(unit, field)


class ExplainedReducer:
    """Wraps an ``SCRE`` engine to also report per-unit scoring/selection detail.

    Composes an ``SCRE`` instance (never mutates or subclasses it) and
    reruns the same public pipeline stages ``SCRE.reduce()`` uses
    internally, purely to keep the intermediate per-unit state that
    ``reduce()`` discards by design -- a UI explaining *why* a sentence
    was kept or dropped needs that detail; a caller just wanting the
    compressed text (the overwhelmingly common case) does not, which is
    why ``reduce()`` itself stays lean.
    """

    def __init__(self, engine: SCRE | None = None):
        self.engine = engine if engine is not None else SCRE()

    def reduce_with_explanation(
        self,
        text: str,
        query: str = DEFAULT_QUERY,
        max_sentences: int = 6,
        context_window: int = 1,
    ) -> dict[str, Any]:
        """Same reduction ``SCRE.reduce()`` would produce, plus per-unit detail.

        Returns a dict with ``context``, ``metadata`` (same shape as
        ``SCRE.reduce()``), ``units`` (one entry per extracted sentence:
        ``index``, ``text``, ``type``, ``score``, ``kept``), and
        ``reasoning_edges`` (deduplicated ``[a, b]`` pairs from the
        reasoning graph, for a graph visualization).
        """
        engine = self.engine
        sentences = segment_text(text, engine.nlp)
        if not sentences:
            return {
                "context": "",
                "metadata": {
                    "reduction_ratio": 0.0, "original_chars": 0, "reduced_chars": 0,
                    "original_sentences": 0, "reduced_sentences": 0,
                    "original_estimated_tokens": 0, "reduced_estimated_tokens": 0,
                },
                "units": [],
                "reasoning_edges": [],
                "prompt_type": "system_prompt",
            }

        original_chars = len(text)
        original_sentences = len(sentences)
        original_estimated_tokens = estimate_tokens(text)

        all_units = build_semantic_state(sentences, engine.extractors)

        # Classified before scoring/selection so the classification can
        # actually change how much gets kept -- see the effective_* below.
        prompt_type = classify_prompt_type(text, all_units, original_sentences)

        knowledge_graph = build_knowledge_graph(all_units)
        reasoning_graph = build_reasoning_graph(all_units, engine.config)
        apply_graph_connectivity(knowledge_graph, all_units)
        apply_reasoning_connectivity(reasoning_graph, all_units)

        query_features = get_query_features(query, engine.nlp)
        idf = compute_idf([DummySentence(u.original_sentence_text) for u in all_units])

        dense_scores: list[float] = []
        if engine.embedder and all_units:
            import numpy as np
            import torch
            from sentence_transformers import util as st_util

            unit_texts = [u.original_sentence_text for u in all_units]
            unit_embs = engine.embedder.encode(unit_texts, convert_to_numpy=True).astype(np.float32)
            query_emb = engine.embedder.encode(query, convert_to_tensor=True)
            unit_embs_tensor = torch.tensor(unit_embs).to(query_emb.device)
            dense_scores = st_util.cos_sim(query_emb, unit_embs_tensor)[0].tolist()

        types_with_labeled_instance = {
            u.type for u in all_units if u.extraction_source == "labeled"
        }
        for idx, unit in enumerate(all_units):
            dense_score = dense_scores[idx] if dense_scores else 0.0
            score_semantic_unit(unit, query_features, idf, dense_score, engine.config, types_with_labeled_instance)

        # A paragraph-style generative/image prompt has no decision-record
        # structure for the reasoning graph to differentiate on -- every
        # sentence is a roughly equally load-bearing visual/descriptive
        # detail, not filler. Dropping any of it risks changing what gets
        # generated, unlike a spec document where a low-priority "fact"
        # really is safe to cut. So for that classification, request
        # (almost) everything: max_sentences covers every sentence, and
        # min_tokens is set high enough that the floor never triggers an
        # early cutoff -- select_semantic_units still caps this at the
        # total token count of positively-scored candidates, so it can't
        # force in anything that scored zero or negative.
        if prompt_type == "unstructured_prompt":
            effective_max_sentences = original_sentences
            effective_min_tokens = original_estimated_tokens
        else:
            effective_max_sentences = max_sentences
            effective_min_tokens = engine.config.min_tokens_default

        selected_units = select_semantic_units(
            units=all_units,
            max_sentences=effective_max_sentences,
            max_tokens=None,
            min_tokens=effective_min_tokens,
            config=engine.config,
            reasoning_graph=reasoning_graph,
        )
        if selected_units:
            if context_window > 0:
                selected_units = expand_adjacent_context(selected_units, all_units, context_window)
            selected_units = expand_reasoning_chains(selected_units, all_units, reasoning_graph)

        selected_units = drop_orphan_headings(selected_units, all_units)
        reduced_text = build_compressed_context(selected_units)

        # By index, not unit identity -- SemanticUnit.__eq__ compares
        # (type, subject, relation, object), so two structurally-identical
        # sentences at different indices must still be reported separately.
        kept_indices = {u.original_sentence_index for u in selected_units}

        units_out = [
            {
                "index": u.original_sentence_index,
                "text": u.original_sentence_text,
                "type": u.type,
                "score": round(u.score, 3),
                "kept": u.original_sentence_index in kept_indices,
                "extraction_source": u.extraction_source,
            }
            for u in sorted(all_units, key=lambda u: u.original_sentence_index)
        ]
        reasoning_edges = sorted({
            tuple(sorted((src, tgt)))
            for src, targets in reasoning_graph.items()
            for tgt in targets
        })

        return {
            "context": reduced_text,
            "metadata": {
                "original_sentences": original_sentences,
                "reduced_sentences": len(selected_units),
                "original_chars": original_chars,
                "reduced_chars": len(reduced_text),
                "original_estimated_tokens": original_estimated_tokens,
                "reduced_estimated_tokens": estimate_tokens(reduced_text),
                "reduction_ratio": round(1 - (len(reduced_text) / max(original_chars, 1)), 4),
            },
            "units": units_out,
            "reasoning_edges": [list(edge) for edge in reasoning_edges],
            "prompt_type": prompt_type,
        }

    def compare_understanding(self, original_text: str, reduced_text: str, model_key: str) -> dict[str, Any]:
        """Ask the same question of the original and reduced content, then judge if the answers match.

        Uses ``scre.scre_answer_engine``'s already-public
        ``answer_with_ollama``/``judge_meaning_retention`` -- this is
        model-comparison logic the library ships, not something added to
        it for this UI.
        """
        model_name = MODEL_NAMES[model_key]
        original_answer = answer_with_ollama(original_text, UNDERSTANDING_QUESTION, model_name=model_name)
        reduced_answer = answer_with_ollama(reduced_text, UNDERSTANDING_QUESTION, model_name=model_name)
        verdict = judge_meaning_retention(UNDERSTANDING_QUESTION, original_answer, reduced_answer, model_name=model_name)
        return {
            "original": original_answer,
            "reduced": reduced_answer,
            "retained": verdict["match"],
            "reason": verdict["reason"],
        }
