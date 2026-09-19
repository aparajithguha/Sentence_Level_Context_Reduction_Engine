"""
query_aware_reducer.py
======================
Core module for the Sentence-level Context Reduction Engine (SCRE).

SCRE is stateless: ``SCRE.reduce(text, query, ...)`` is the single entry
point, and it is a self-contained, in-memory computation over one
document/query pair -- nothing is written to disk or retained after the
call returns. There is no document cache, no ``document_id``, and no
persistence layer, by design: SCRE is meant to sit inline on a live prompt
path, where each call is one prompt and the reduced result is handed to the
LLM immediately.

This module is a thin facade: the ``SCRE`` class owns configuration
(``nlp``, ``embedder``, ``config``, ``extractors``) and orchestrates calls
into single-responsibility modules that hold the actual pipeline logic:

- ``scre.utils``      -- stateless text/token helpers
- ``scre.units``       -- ``SemanticUnit``/``WorkflowUnit`` data model
- ``scre.extraction``  -- sentence segmentation + extractor-chain classification
- ``scre.graph``       -- ``KnowledgeGraph`` + reasoning-graph construction
- ``scre.scoring``     -- query feature extraction + hybrid unit scoring
- ``scre.selection``   -- token-budgeted selection + expansion
- ``scre.assembly``    -- final compressed-context rendering

System/agent prompts have their own path, ``SCRE.reduce_prompt`` (see
``scre.prompt_reducer``): it keeps every instruction and removes only
repeated instructions, surplus examples and decoration, and needs neither
spaCy nor the embedder.

``reduce`` runs in two halves:

1. **Analysis** (``SCRE._analyze``, also exposed query-independently via
   ``SCRE.analyze`` for introspection/visualization): segments raw text into
   sentences, extracts structured ``SemanticUnit`` objects via pluggable
   extractor strategies, and builds a Knowledge Graph and a Reasoning Graph
   -- all held as plain in-memory objects for the duration of the call.

2. **Scoring & selection**: computes dense embeddings, scores every unit
   using a hybrid signal (TF-IDF lexical overlap, dense cosine similarity,
   entity/noun-chunk match, graph connectivity, intent type-match, and
   hard-identifier boost), selects the highest-scoring non-duplicate units,
   expands the selection via adjacent context and multi-hop reasoning
   chains, and assembles a compressed context string.

The spaCy pipeline and sentence-transformer model are process-wide
singletons (see ``scre.models``) -- loaded once and shared across every
``SCRE`` instance, since loading either is expensive relative to a single
reduction.

Public API
----------
- ``SCRE``       – Main engine class (``reduce``, ``reduce_prompt``, ``analyze``).
- ``PromptReducer`` – Prompt-mode reducer used by ``reduce_prompt`` (``scre.prompt_reducer``).
- ``SemanticUnit``  – Atomic unit of extracted meaning (``scre.units``).
- ``WorkflowUnit``  – Specialised unit for ordered procedural steps (``scre.units``).
- ``UnitExtractorStrategy`` – Abstract base for extraction plug-ins (``scre.extraction``).
- ``RegexKeyValueExtractor`` – Regex-based extraction strategy (``scre.extraction``).
- ``DefaultNLPExtractor``    – SpaCy-powered fallback extraction strategy (``scre.extraction``).
- ``KnowledgeGraph``            – Typed-triple knowledge graph (``scre.graph``).
- ``build_compressed_context`` – Assembles final context string from units (``scre.assembly``).
- ``select_semantic_units``    – Ranks and selects units within token budget (``scre.selection``).

All of the above remain importable from ``scre.query_aware_reducer`` (this
module re-imports them for that purpose) even though their implementations
now live in the modules listed above.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

try:
    import numpy as np
except ImportError:  # only the embedder path needs numpy; prompt mode needs neither
    np = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    from sentence_transformers import util
except ImportError:
    util = None

try:
    from . import models as _models
    from .config import ScoringConfig, PromptConfig
    from .prompt_reducer import PromptReducer
    from .utils import DummySentence, compute_idf, estimate_tokens
    from .units import SemanticUnit, WorkflowUnit
    from .extraction import (
        UnitExtractorStrategy,
        RegexKeyValueExtractor,
        DefaultNLPExtractor,
        segment_text,
        build_semantic_state,
    )
    from .graph import (
        KnowledgeGraph,
        build_knowledge_graph,
        apply_graph_connectivity,
        apply_reasoning_connectivity,
        build_reasoning_graph,
    )
    from .scoring import get_query_features, score_semantic_unit
    from .selection import select_semantic_units, expand_adjacent_context, expand_reasoning_chains
    from .assembly import build_compressed_context, drop_orphan_headings
except ImportError:
    import models as _models
    from config import ScoringConfig, PromptConfig
    from prompt_reducer import PromptReducer
    from utils import DummySentence, compute_idf, estimate_tokens
    from units import SemanticUnit, WorkflowUnit
    from extraction import (
        UnitExtractorStrategy,
        RegexKeyValueExtractor,
        DefaultNLPExtractor,
        segment_text,
        build_semantic_state,
    )
    from graph import (
        KnowledgeGraph,
        build_knowledge_graph,
        apply_graph_connectivity,
        apply_reasoning_connectivity,
        build_reasoning_graph,
    )
    from scoring import get_query_features, score_semantic_unit
    from selection import select_semantic_units, expand_adjacent_context, expand_reasoning_chains
    from assembly import build_compressed_context, drop_orphan_headings

# Enforce strict offline mode now that the model is cached.
# This prevents HF Hub update checks and silences token warnings.
os.environ["HF_HUB_OFFLINE"] = "1"

# Explicit re-export surface: these are imported above only to satisfy the
# "Public API" contract this module's docstring has always documented, not
# because facade code below calls all of them directly (WorkflowUnit isn't).
# Declaring __all__ makes that intentional, and keeps linters from flagging
# WorkflowUnit as an unused import.
__all__ = [
    "SCRE",
    "SemanticUnit",
    "WorkflowUnit",
    "UnitExtractorStrategy",
    "RegexKeyValueExtractor",
    "DefaultNLPExtractor",
    "KnowledgeGraph",
    "PromptReducer",
    "PromptConfig",
    "build_compressed_context",
    "select_semantic_units",
]


class SCRE:
    """
    Sentence-level Context Reduction Engine.
    A library class to reduce document context based on a query.
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        extractors: list[UnitExtractorStrategy] | None = None,
        nlp: Any = None,
        embedder: Any = None,
        config: ScoringConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ):
        """Initialise the SCRE engine.

        SCRE holds no persistent state and no database connection — each
        call to ``reduce`` is a self-contained, in-memory computation over
        one document/query pair. The spaCy pipeline and sentence-transformer
        model are process-wide singletons (see ``scre.models``): loading
        them is expensive (seconds), so they are loaded once per process and
        shared across every ``SCRE`` instance rather than reloaded per call.

        Args:
            model: Name of the spaCy model to load (e.g. ``"en_core_web_sm"``).
                Pass an empty string or ``None`` to disable NLP-based
                extraction and fall back to regex-only mode. Ignored if
                ``nlp`` is provided.
            extractors: An explicit list of ``UnitExtractorStrategy`` objects
                to use instead of the built-in default chain.  When ``None``
                the default chain is:

                1. ``RegexKeyValueExtractor`` for SDLC-tagged documents.
                2. ``RegexKeyValueExtractor`` for Markdown key-value pairs.
                3. ``DefaultNLPExtractor`` (only when spaCy is available).
            nlp: An already-loaded spaCy ``Language`` object to reuse instead
                of fetching one from the shared cache. Primarily for tests
                and callers with unusual model requirements.
            embedder: An already-loaded ``SentenceTransformer`` to reuse
                instead of fetching one from the shared cache.
            config: Scoring/graph/selection thresholds (see ``scre.config.
                ScoringConfig``). Defaults to ``ScoringConfig()`` — the same
                values previously hardcoded inline.
            prompt_config: Thresholds for ``reduce_prompt`` (see
                ``scre.config.PromptConfig``). Defaults to ``PromptConfig()``.
        """
        if nlp is not None:
            self.nlp = nlp
        elif model:
            self.nlp = _models.get_nlp(model)
        else:
            self.nlp = None

        # Loaded on first use, so prompt mode (``reduce_prompt``) never pays for the embedding model.
        # ``embedder=False`` disables dense scoring outright.
        self._embedder = embedder
        self._spacy_missing = bool(model) and nlp is None and self.nlp is None
        self._warned_spacy = False
        self.config = config if config is not None else ScoringConfig()
        self._prompt_reducer = PromptReducer(prompt_config)

        self.content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}

        if extractors is None:
            self.extractors = [
                # Custom SDLC Benchmark configuration decoupled to a strategy
                RegexKeyValueExtractor(r"^(?:-\s*)?\[([A-Z]+)-\d+\]\s*([^:]+):\s*(.*)", {
                    "FACT": "fact", "DEC": "decision", "CON": "constraint",
                    "TSK": "task", "WRK": "workflow", "REA": "reason", "STC": "state"
                }),
                # Common Markdown Configuration Strategy
                RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
            ]
            if self.nlp is not None:
                self.extractors.append(DefaultNLPExtractor())

        else:
            self.extractors = extractors

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = _models.get_embedder()
        return self._embedder

    @embedder.setter
    def embedder(self, value: Any) -> None:
        self._embedder = value

    def reduce_prompt(
        self, text: str, task: str | None = None, drop_modules: set[int] | list[int] | None = None,
        selector: Any = None,
    ) -> dict[str, Any]:
        """Reduce a system/agent prompt while keeping every instruction.

        Unlike ``reduce``, this selects no sentences by relevance: the prompt
        is split into structural blocks (sections, lists, examples, code/JSON,
        tag pairs) and only redundant blocks are removed -- repeated
        instructions, examples beyond the first few of each group, horizontal
        rules, and the headings/tags left empty by those removals. Kept
        blocks are emitted word for word in their original order; code, JSON
        and tool definitions are never touched. Runs no model.

        Given a ``task``, sections the task cannot use are also dropped
        whole; ``drop_modules`` (ids from ``PromptReducer.outline``) lets the
        caller choose them instead, with the same guards, and ``selector`` (for
        example ``scre.prompt_select.llm_selector(your_ask_function)``) lets an
        LLM you supply choose them, falling back to a safe reduction on failure.

        Returns:
            ``{"context": str, "metadata": {...}, "blocks": [...]}`` -- see
            ``scre.prompt_reducer.PromptReducer.reduce``.
        """
        return self._prompt_reducer.reduce(text, task=task, drop_modules=drop_modules, selector=selector)

    def _analyze(self, text: str) -> tuple[list[SemanticUnit], KnowledgeGraph, dict[int, set[int]], int, int, int]:
        """Segment, classify, and graph ``text`` — the query-independent half of ``reduce``.

        Shared by ``reduce`` (which scores the result against a query) and
        ``analyze`` (which exposes it directly for introspection/visualization,
        e.g. a reasoning-graph explorer UI) so both stay backed by exactly one
        segmentation/extraction/graph-construction implementation.

        Returns:
            A tuple of ``(units, knowledge_graph, reasoning_graph,
            original_chars, original_sentences, original_estimated_tokens)``.
            ``units`` is empty when ``text`` contains no extractable sentences.
        """
        sentences = segment_text(text, self.nlp)

        if not sentences:
            return [], KnowledgeGraph(), {}, len(text), 0, estimate_tokens(text)

        original_chars = len(text)
        original_sentences = len(sentences)
        original_estimated_tokens = estimate_tokens(text)

        all_extracted_units = build_semantic_state(sentences, self.extractors)

        knowledge_graph = build_knowledge_graph(all_extracted_units)
        reasoning_graph = build_reasoning_graph(all_extracted_units, self.config)

        return all_extracted_units, knowledge_graph, reasoning_graph, original_chars, original_sentences, original_estimated_tokens

    def analyze(self, text: str) -> dict[str, Any]:
        """Expose extracted units and reasoning-graph edges for ``text``, without scoring against a query.

        For introspection and visualization only (e.g. a reasoning-graph
        explorer UI) — ``reduce`` is the entry point for actual retrieval and
        does not need this. Like ``reduce``, this is a self-contained,
        in-memory computation; nothing is persisted.

        Returns:
            A dict with two keys:

            * ``"units"`` (*list[dict]*) — one entry per extracted unit, each
              with ``sentence_index``, ``text``, ``unit_type``, and
              ``render_text`` (a JSON string for workflow units, otherwise
              the same as ``text``).
            * ``"reasoning_edges"`` (*list[tuple[int, int]]*) — directed
              ``(source_idx, target_idx)`` pairs from the reasoning graph
              (stored symmetrically, so each connected pair appears twice,
              once in each direction).
        """
        all_extracted_units, _knowledge_graph, reasoning_graph, *_ = self._analyze(text)
        units = [
            {
                "sentence_index": u.original_sentence_index,
                "text": u.original_sentence_text,
                "unit_type": u.type,
                "render_text": getattr(u, "render_text", u.original_sentence_text),
            }
            for u in all_extracted_units
        ]
        reasoning_edges = [
            (src, tgt) for src, targets in reasoning_graph.items() for tgt in targets
        ]
        return {"units": units, "reasoning_edges": reasoning_edges}

    def reduce(
        self,
        text: str,
        query: str,
        max_sentences: int = 4,
        max_tokens: int | None = None,
        context_window: int = 1,
        min_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Reduce ``text`` to the subset most relevant to ``query``.

        This is the engine's only entry point. It is a single, self-contained,
        in-memory computation over one document/query pair — nothing is
        written to disk or retained after the call returns. There is no
        document cache, no ``document_id``, and no persistence layer: SCRE is
        designed to sit inline on a live prompt path, where each call is one
        prompt, and the reduced result is handed to the LLM immediately.

        The pipeline:

        1. **Pre-processing**: Merges split ``Label:\n Value`` pairs that would
           otherwise confuse sentence tokenisers.
        2. **Sentence segmentation** (``extraction.segment_text``): Splits
           paragraphs into sentences via spaCy (or a simple regex fallback).
           List items, headers, and ``Label: value`` lines are kept intact to
           preserve document structure and category boundaries.
        3. **Semantic classification** (``extraction.build_semantic_state``):
           Runs the extractor chain over each sentence to produce
           ``SemanticUnit`` objects, detects ordered workflows, and tracks
           Markdown headings for context-header injection.
        4. **Graph construction**: Builds a Knowledge Graph
           (``graph.build_knowledge_graph``) and a Reasoning Graph
           (``graph.build_reasoning_graph``) for multi-hop retrieval — held as
           plain in-memory dicts for the duration of this call only.
        5. **Embedding**: Encodes all unit texts with the sentence-transformer
           model (if available).
        6. **Scoring** (``scoring.score_semantic_unit``): Scores every unit
           using a hybrid signal (lexical IDF, dense cosine, entity match,
           knowledge-graph connectivity, reasoning-graph connectivity,
           intent type-match, hard-identifier boost).
           A heuristically-typed unit's intent-match credit is discounted
           when a labeled (document-asserted) unit of the same type exists
           elsewhere in the document -- see ``types_with_labeled_instance``.
        7. **Selection** (``selection.select_semantic_units``): Selects the
           top-scoring non-duplicate units within the token budget.
        8. **Expansion**: Expands the selection with adjacent sentences
           (``selection.expand_adjacent_context``) and multi-hop reasoning
           chain nodes (``selection.expand_reasoning_chains``).
        9. **Assembly** (``assembly.build_compressed_context``): Assembles
           and returns the compressed context string.

        Args:
            text: Raw document/prompt text to reduce.
            query: The natural-language question or search query.
            max_sentences: Maximum number of ``SemanticUnit`` objects to
                include before context-window and reasoning-chain expansion.
                Defaults to ``4``.
            max_tokens: Hard upper bound on the token count of the compressed
                context.  ``None`` means no hard limit.
            context_window: Number of adjacent sentences to include around
                each selected unit for coreference resolution.  Set to
                ``0`` to disable.  Defaults to ``1``.
            min_tokens: Minimum token count that must be reached before the
                ``max_sentences`` cap is enforced.  Prevents premature
                truncation of short selections.  Capped internally at the
                total token count of all positively-scored candidates, so it
                cannot force irrelevant content into a short selection just
                to hit an arbitrary floor.  Defaults to ``self.config.min_tokens_default``
                (``250``) when not given.

        Returns:
            A dict with two keys:

            * ``"context"`` (*str*) – The compressed, query-aligned context.
            * ``"metadata"`` (*dict*) – Reduction statistics including
              ``original_sentences``, ``reduced_sentences``,
              ``original_chars``, ``reduced_chars``,
              ``original_estimated_tokens``, ``reduced_estimated_tokens``,
              and ``reduction_ratio``.
        """
        if self._spacy_missing and not self._warned_spacy:
            self._warned_spacy = True
            warnings.warn("spaCy is not installed, so document mode runs in regex-only mode. "
                          "Install it with: pip install 'scre[document]'", stacklevel=2)
        all_extracted_units, knowledge_graph, reasoning_graph, original_chars, original_sentences, original_estimated_tokens = self._analyze(text)

        if not all_extracted_units:
            return {"context": "", "metadata": {"reduction_ratio": 0.0, "original_chars": 0, "reduced_chars": 0}}

        # Dense Embeddings
        unit_embs = None
        if self.embedder and all_extracted_units:
            unit_texts = [u.original_sentence_text for u in all_extracted_units]
            unit_embs = self.embedder.encode(unit_texts, convert_to_numpy=True).astype(np.float32)

        apply_graph_connectivity(knowledge_graph, all_extracted_units)
        apply_reasoning_connectivity(reasoning_graph, all_extracted_units)

        query_features = get_query_features(query, self.nlp)
        sentences_for_idf = [DummySentence(u.original_sentence_text) for u in all_extracted_units]
        idf = compute_idf(sentences_for_idf)

        dense_scores = []
        if self.embedder and unit_embs is not None and len(unit_embs) > 0:
            import torch
            query_emb = self.embedder.encode(query, convert_to_tensor=True)
            unit_embs_tensor = torch.tensor(unit_embs).to(query_emb.device)
            dense_scores = util.cos_sim(query_emb, unit_embs_tensor)[0].tolist()

        types_with_labeled_instance = {
            u.type for u in all_extracted_units if u.extraction_source == "labeled"
        }
        for idx, unit in enumerate(all_extracted_units):
            dense_score = dense_scores[idx] if dense_scores else 0.0
            score_semantic_unit(unit, query_features, idf, dense_score, self.config, types_with_labeled_instance)

        selected_units = select_semantic_units(
            units=all_extracted_units,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            min_tokens=min_tokens if min_tokens is not None else self.config.min_tokens_default,
            config=self.config,
            reasoning_graph=reasoning_graph,
        )

        if selected_units:
            if context_window > 0:
                selected_units = expand_adjacent_context(selected_units, all_extracted_units, context_window)
            selected_units = expand_reasoning_chains(selected_units, all_extracted_units, reasoning_graph)

        # A heading pulled in by adjacency but with nothing selected under it is noise, not context.
        selected_units = drop_orphan_headings(selected_units, all_extracted_units)
        reduced_text = build_compressed_context(selected_units)
        return {
            "context": reduced_text,
            "metadata": {
                "original_sentences": original_sentences,
                "reduced_sentences": len(selected_units),
                "selected_sentence_count": len(selected_units),
                "original_chars": original_chars,
                "reduced_chars": len(reduced_text),
                "original_estimated_tokens": original_estimated_tokens,
                "reduced_estimated_tokens": estimate_tokens(reduced_text),
                "reduction_ratio": round(1 - (len(reduced_text) / max(original_chars, 1)), 4)
            }
        }
