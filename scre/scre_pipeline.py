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
    """
    Build a reduced, question-aligned context packet
    that can be forwarded to an answer model.
    """

    # Initialize the engine (in a real library, you'd do this once)
    engine = SCRE()

    # Phase 1 & 2: Retrieval & Compression
    reduction = engine.reduce(
        text=document_text,
        query=user_question,
        max_sentences=max_sentences,
        context_window=context_window,
    )

    # Step 2: Use the reduced context to get an answer
    answer_result = answer_question(
        reduced_context=reduction["context"],
        user_question=user_question,
        mode=answer_mode,
        model_name=model_name or "gemma4:e2b",
    )

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
