"""
scre_pipeline.py
================
High-level convenience pipeline that wires ``SCRE`` (reduction) and
``answer_question`` (answer generation) into a single callable.

This module is the simplest entry-point for end-to-end usage::

    from scre.scre_pipeline import run_scre

    result = run_scre(
        document_text="...",
        user_question="Who manages Project Phoenix?",
        answer_mode="ollama",
    )
    print(result["answer"])

``SCRE`` is stateless by design — each call constructs a fresh, cheap engine
instance (the underlying spaCy/embedder models are shared, process-wide
singletons, so this does not reload them) and performs one self-contained
reduction. There is no ingest/retrieve split and no document cache to manage.
"""
import json
from typing import Any

try:
    from .query_aware_reducer import SCRE
    from .scre_answer_engine import answer_question
except ImportError:
    from query_aware_reducer import SCRE
    from scre_answer_engine import answer_question


def run_scre(
    document_text: str,
    user_question: str,
    max_sentences: int = 4,
    answer_mode: str = "ollama",
    context_window: int = 1,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Reduce a document to the context most relevant to a question, then answer it.

    This is a convenience wrapper that:

    1. Constructs a fresh ``SCRE`` engine (cheap — model loading is shared
       across instances via ``scre.models``, not repeated per call).
    2. Calls ``SCRE.reduce`` to compress ``document_text`` down to the
       compressed, query-aligned context.
    3. Passes the compressed context to ``answer_question`` to derive an
       answer in the requested mode.

    Args:
        document_text: The full raw text of the source document.
        user_question: The natural-language question to answer.
        max_sentences: Maximum number of semantic units to include in the
            compressed context (before context-window expansion).  Defaults
            to ``4``.
        answer_mode: Answer backend to use.  ``"ollama"`` calls a local
            Ollama LLM; ``"extractive"`` uses offline span extraction.
            Defaults to ``"ollama"``.
        context_window: Number of adjacent sentences to include around each
            selected unit for coreference resolution.  Defaults to ``1``.
        model_name: Ollama model name; only used when ``answer_mode="ollama"``.
            ``None`` (default) defers to ``answer_question``'s own default
            (``scre_answer_engine.DEFAULT_MODEL_NAME``) rather than
            duplicating that literal here.

    Returns:
        A dict with the following keys:

        * ``"question"`` (*str*): Echo of the original question.
        * ``"reduced_context"`` (*str*): The compressed context used to
          derive the answer.
        * ``"metrics"`` (*dict*): Reduction statistics from
          ``SCRE.reduce`` (chars, tokens, reduction ratio, etc.).
        * ``"answer"`` (*dict*): Answer payload from ``answer_question``,
          containing ``"mode"`` and ``"answer"`` keys.
    """

    # Cheap: model loading is a shared, process-wide singleton (see scre.models),
    # so constructing a fresh engine per call does not reload spaCy/the embedder.
    engine = SCRE()

    # Phase 1 & 2: Retrieval & Compression
    reduction = engine.reduce(
        text=document_text,
        query=user_question,
        max_sentences=max_sentences,
        context_window=context_window,
    )

    # Step 2: Use the reduced context to get an answer. Omit model_name
    # entirely when unset so answer_question's own DEFAULT_MODEL_NAME
    # governs -- avoids duplicating that literal here.
    answer_kwargs: dict[str, Any] = {
        "reduced_context": reduction["context"],
        "user_question": user_question,
        "mode": answer_mode,
    }
    if model_name is not None:
        answer_kwargs["model_name"] = model_name
    answer_result = answer_question(**answer_kwargs)

    return {
        "question": user_question,
        "reduced_context": reduction["context"],
        "metrics": reduction["metadata"],
        "answer": answer_result,
    }


if __name__ == "__main__":
    text = """
    John owns Project Phoenix.
    Sarah manages Project Phoenix.
    Mike tests Project Phoenix.
    David deploys Project Phoenix.
    Lisa audits Project Phoenix.
    Emma documents Project Phoenix.
    """

    question = "Who manages Project Phoenix?"

    result = run_scre(text, question)

    print("\n===== SCRE OUTPUT =====\n")
    print(json.dumps(result, indent=4))
