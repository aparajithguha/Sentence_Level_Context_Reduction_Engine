"""
scre_answer_engine.py
=====================
Answer-generation layer for the Sentence-level Context Reduction Engine.

This module operates on the *compressed context* produced by ``SCRE.retrieve``
and derives a final answer to the user's question via one of two modes:

* **Extractive** (default, offline): Uses spaCy dependency-parsing and NER
  to identify the most relevant span within the compressed context.  Handles
  WH-question types (who / what / where / when / why) with tailored
  extraction heuristics.
* **Ollama** (generative): Forwards the compressed context and question to a
  locally running Ollama LLM (default model ``qwen3:4b-instruct-2507-q4_K_M``
  -- chosen over the previous default, ``gemma4:e2b``, after an empirical
  side-by-side: identical answer correctness on the same questions, but
  2-3x lower latency and under half the disk footprint, since gemma4:e2b's
  extra parameters go to vision/audio capability this text-only task never
  uses) via a structured prompt and returns the model's free-form answer.

Also provides ``judge_meaning_retention``: a third LLM call that checks
whether a reduced-context answer preserves the meaning of the full-context
answer, for evaluating whether SCRE's reduction actually retains meaning
rather than just shrinking text.

Public API
----------
- ``answer_question``    – Unified entry-point; dispatches to the chosen mode.
- ``extractive_answer``  – Offline span-extraction fallback.
- ``answer_with_ollama`` – Generative answer via a local Ollama LLM.
- ``build_answer_prompt``– Constructs the structured LLM prompt.
- ``judge_meaning_retention`` – LLM-judged full-vs-reduced answer equivalence.
- ``build_judge_prompt`` – Constructs the structured judge prompt.
- ``score_line_for_question`` – Scores a context line against the question.
- ``content_terms``      – Extracts lemmatised content tokens from text.
"""
from __future__ import annotations

import re
from typing import Any

try:
    from . import models as _models
    from .units import strip_json_workflows
except ImportError:
    import models as _models
    from units import strip_json_workflows


# Shared, process-wide spaCy pipeline (see scre.models) -- reused rather
# than loaded independently, so an SCRE() instance and this module's
# extractive-answer path never pay to load the model twice.
NLP = _models.get_nlp()

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency at runtime
    ollama = None


DEFAULT_MODEL_NAME = "qwen3:4b-instruct-2507-q4_K_M"


def normalize_text(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space and strip ends.

    Args:
        text: Raw input string.

    Returns:
        Normalised string.
    """
    return re.sub(r"\s+", " ", text).strip()


def content_terms(text: str) -> list[str]:
    """Extract lemmatised content tokens from a text string.

    Stop words, punctuation, and numeric tokens are excluded.  The result
    is suitable for lightweight lexical overlap comparisons.

    Args:
        text: Input text to tokenise and lemmatise.

    Returns:
        A list of lowercase lemma strings for each content token.
    """
    doc = NLP(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
    ]


def score_line_for_question(line: str, user_question: str) -> float:
    """Score a single context line against a user question.

    The score combines three signals:

    * **Lexical overlap**: Count of lemmatised content tokens shared between
      the question and the line.
    * **Entity bonus** (+1.5 per entity): Named entities in the line that
      also appear (as substrings) in the question.
    * **Root verb bonus** (+3.0 exact / +1.5 partial): Whether the syntactic
      root of the question also appears as the root of the line (or anywhere
      in its token set).

    Args:
        line: A single line from the compressed context.
        user_question: The original user question.

    Returns:
        A non-negative float; higher means more relevant.
    """
    question_doc = NLP(user_question)
    query_terms = {
        token.lemma_.lower()
        for token in question_doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
    }
    query_root = question_doc[:].root.lemma_.lower().strip()
    line_doc = NLP(line)
    line_terms = {
        token.lemma_.lower()
        for token in line_doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
    }
    lexical_overlap = len(query_terms & line_terms)

    entity_bonus = 0
    question_lower = user_question.lower()

    for ent in line_doc.ents:
        if ent.text.lower() in question_lower:
            entity_bonus += 1

    root_bonus = 0
    line_root = line_doc[:].root.lemma_.lower().strip()

    if query_root:
        if query_root == line_root:
            root_bonus = 3.0
        elif query_root in line_terms:
            root_bonus = 1.5

    return lexical_overlap + (entity_bonus * 1.5) + root_bonus


def build_answer_prompt(
    reduced_context: str,
    user_question: str,
) -> str:
    """Build the structured prompt that is sent to the Ollama LLM.

    The prompt instructs the model to answer strictly from the provided
    context and to respond with a fixed fallback phrase when the context
    does not contain the answer.

    Args:
        reduced_context: The compressed context string produced by SCRE.
        user_question: The original user question.

    Returns:
        A formatted prompt string ready to be sent as a user message.
    """
    return f"""Answer the user's question using only the provided context.

Note: The context may contain structured JSON objects representing procedural workflows or lists. Treat these JSON blocks as valid factual evidence.

If the context does not contain the answer, reply exactly with:
I cannot determine that from the context.

Context:
{reduced_context}

Question:
{user_question}
"""


def _causal_or_labeled_answer(ranked_lines: list[str]) -> str | None:
    """Find a causal/explanatory answer among ``ranked_lines``, for "why"/"how" questions.

    Tries, in order:

    1. The clause following the literal word "because" in the top-ranked
       line that contains one.
    2. The value of a structured ``Reason:`` or ``Outcome:`` labeled line --
       SCRE's own extraction convention for causal/explanatory content
       (see ``RegexKeyValueExtractor``), which pure lexical "because"
       matching is otherwise blind to: a document that states its reasoning
       via an explicit ``Reason:`` field rather than the word "because"
       would previously fall straight through to the generic top-ranked-
       line fallback, ignoring the actual reason sitting right there.

    Returns ``None`` if neither is found, so the caller can apply its own
    final fallback.
    """
    for line in ranked_lines:
        lowered = line.lower()
        if "because" in lowered:
            start = lowered.index("because") + len("because")
            causal_clause = line[start:].strip().rstrip(".")
            return causal_clause.split(",", 1)[0].strip()

    for line in ranked_lines:
        match = re.match(r'^(?:reason|outcome)\s*:\s*(.+)', line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _render_workflow_as_text(workflow: dict) -> str:
    """Render a workflow dict back into a single plain-text line.

    Lets a preserved workflow participate in the same line-ranking and
    span-extraction logic as ordinary prose, instead of being invisible --
    without this, a workflow's JSON block (see ``units.strip_json_workflows``)
    would simply be missing from ``context_lines``, and any question whose
    answer lives inside it (e.g. "what are the stages of X") would silently
    fall back to whatever unrelated prose line ranks highest instead.
    """
    name = workflow.get("name", "").strip()
    steps_text = " ".join(s.strip() for s in workflow.get("steps", []))
    return f"{name}: {steps_text}" if name else steps_text


def extractive_answer(
    reduced_context: str,
    user_question: str,
) -> str:
    """Lightweight offline extractive answer using spaCy parsing heuristics.

    Selects the single best span from the compressed context without calling
    any external model.  Suitable for demo, evaluation, and air-gapped
    environments.

    A preserved workflow is rendered into the reduced context as a JSON
    block (see ``units.WorkflowUnit.render_text``), which ordinary line-
    splitting and dependency parsing can't read at all -- it would shred
    into meaningless fragments (``'{'``, ``'"steps": ['``, ...). Any such
    block is extracted and re-rendered as a single plain-text line (see
    ``_render_workflow_as_text``) before the ranking/extraction logic below
    ever runs, so a workflow's content is visible on equal footing with
    ordinary prose lines.

    Strategy by WH-word:

    * **who**: Returns the subject noun phrase or the first PERSON/ORG entity
      in the top-ranked line.
    * **what / where / when**: Returns the direct-object, prepositional-object,
      attribute, or open predicative complement span from the top-ranked line.
    * **why / how**: Extracts the clause following the word "because" in the
      top-ranked line, falling back to a structured ``Reason:``/``Outcome:``
      labeled line, then to the top-ranked line itself (see
      ``_causal_or_labeled_answer``). "how" is treated the same as "why"
      here because this engine's typical "how does X satisfy Y"-style
      question is asking for the same causal/explanatory content a "why"
      question would want, not a step-by-step procedure (that's what a
      workflow unit's own steps are for).
    * **other**: Returns the highest-scoring line in full.

    Lines are ranked by ``score_line_for_question`` before extraction.

    Args:
        reduced_context: Compressed context string produced by ``SCRE.retrieve``.
        user_question: The original user question.

    Returns:
        An answer string extracted from the context, or the fallback message
        ``"I cannot determine that from the context."`` if the context is empty.
    """

    prose_text, workflows = strip_json_workflows(reduced_context)
    workflow_lines = {_render_workflow_as_text(wf) for wf in workflows}
    context_lines = [
        re.sub(r"^\[S\d+\]\s*", "", line).strip()
        for line in prose_text.splitlines()
        if line.strip()
    ] + list(workflow_lines)

    if not context_lines:
        return "I cannot determine that from the context."

    question_doc = NLP(user_question)
    wh_word = next(
        (
            token.lower_
            for token in question_doc
            if token.lower_ in {"who", "what", "where", "when", "why", "how"}
        ),
        None,
    )
    ranked_lines = sorted(
        context_lines,
        key=lambda line: score_line_for_question(line, user_question),
        reverse=True,
    )

    if wh_word == "who":
        for line in ranked_lines:
            sentence = NLP(line)
            subject_phrase = next(
                (
                    " ".join(token.text for token in token.subtree).strip()
                    for token in sentence
                    if token.dep_ in {"nsubj", "nsubjpass"}
                ),
                None,
            )
            if subject_phrase:
                return subject_phrase

            for ent in sentence.ents:
                if ent.label_ in {"PERSON", "ORG"}:
                    return ent.text

    if wh_word in {"what", "where", "when"}:
        # A workflow-rendered line ("name: step1 step2 ...") isn't a real
        # sentence -- the colon and run-together steps regularly mislead
        # the dependency parser into grabbing a fragment like a list number
        # (e.g. "1") instead of the actual content. Return it whole, the
        # same way the why/how path already falls back to the full line
        # when its own targeted extraction finds nothing.
        if ranked_lines[0] in workflow_lines:
            return ranked_lines[0]
        for line in ranked_lines:
            sentence = NLP(line)
            answer_span = next(
                (
                    token.subtree
                    for token in sentence
                    if token.dep_ in {"dobj", "pobj", "attr", "oprd"}
                ),
                None,
            )
            if answer_span:
                return " ".join(token.text for token in answer_span).strip()

    if wh_word in {"why", "how"}:
        answer = _causal_or_labeled_answer(ranked_lines)
        if answer is not None:
            return answer
        return ranked_lines[0]

    return ranked_lines[0]


def answer_with_ollama(
    reduced_context: str,
    user_question: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Generate an answer using a locally running Ollama LLM.

    Constructs the prompt via ``build_answer_prompt`` and forwards it to
    the specified Ollama model.  The context window is explicitly expanded
    to 8 192 tokens to prevent silent truncation of long contexts.

    Args:
        reduced_context: Compressed context string from SCRE.
        user_question: The original user question.
        model_name: Name of the Ollama model to query.  Defaults to
            ``DEFAULT_MODEL_NAME`` (``"qwen3:4b-instruct-2507-q4_K_M"``).

    Returns:
        The model's answer as a stripped string.

    Raises:
        RuntimeError: If the ``ollama`` package is not installed.
    """
    if ollama is None:
        raise RuntimeError("ollama package is not installed.")

    prompt = build_answer_prompt(
        reduced_context=reduced_context,
        user_question=user_question,
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "num_ctx": 8192,  # Expand context window to avoid silent truncation
        }
    )

    return response["message"]["content"].strip()


def build_judge_prompt(question: str, full_answer: str, reduced_answer: str) -> str:
    """Build the structured prompt that asks an LLM to judge meaning retention.

    Compares two independently-generated answers to the same question --
    one from the full, unreduced document, one from SCRE's reduced context
    -- and asks whether they convey the same meaning. This is the actual
    test of whether reduction retains meaning, as opposed to comparing raw
    text overlap between the two contexts.

    Args:
        question: The original user question both answers were generated for.
        full_answer: The answer generated from the full, unreduced context.
        reduced_answer: The answer generated from SCRE's reduced context.

    Returns:
        A formatted prompt string ready to be sent as a user message.
    """
    return f"""You are judging whether two answers to the same question convey the same meaning.

Question:
{question}

Answer A (from the full document):
{full_answer}

Answer B (from a compressed version of the document):
{reduced_answer}

Do Answer A and Answer B convey the same meaning, even if worded differently or B is less detailed? Ignore differences in phrasing, verbosity, or formatting -- judge only whether the core factual content matches. If either answer is "I cannot determine that from the context" and the other is not, that is a MISMATCH.

Respond in exactly this format, nothing else:
VERDICT: MATCH or MISMATCH
REASON: one short sentence explaining why
"""


def judge_meaning_retention(
    question: str,
    full_answer: str,
    reduced_answer: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Ask an LLM whether a reduced-context answer retains the meaning of the full-context answer.

    This is the LLM-graded counterpart to lexical heuristics like MRS
    (``tests/scre_eval.py``'s ``calculate_mrs``): instead of checking
    whether expected keywords survive in the reduced *context*, it checks
    whether an LLM's *understanding* of the reduced context -- as expressed
    in its answer -- still matches its understanding of the full context.

    Args:
        question: The original user question.
        full_answer: The answer generated from the full, unreduced context
            (see ``answer_with_ollama``).
        reduced_answer: The answer generated from SCRE's reduced context.
        model_name: Name of the Ollama model to use as judge.  Defaults to
            ``DEFAULT_MODEL_NAME``.

    Returns:
        A dict with keys:

        * ``"match"`` (*bool*): Whether the judge ruled the two answers
          equivalent in meaning.  ``False`` if the judge's response could
          not be parsed.
        * ``"reason"`` (*str*): The judge's one-line explanation, or a
          parse-failure note.
        * ``"raw"`` (*str*): The judge's raw, unparsed response.

    Raises:
        RuntimeError: If the ``ollama`` package is not installed.
    """
    if ollama is None:
        raise RuntimeError("ollama package is not installed.")

    prompt = build_judge_prompt(question, full_answer, reduced_answer)

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},
    )
    raw = response["message"]["content"].strip()

    verdict_match = re.search(r"VERDICT:\s*(MATCH|MISMATCH)", raw, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)

    if not verdict_match:
        return {"match": False, "reason": "Could not parse judge response.", "raw": raw}

    return {
        "match": verdict_match.group(1).upper() == "MATCH",
        "reason": reason_match.group(1).strip() if reason_match else "",
        "raw": raw,
    }


def answer_question(
    reduced_context: str,
    user_question: str,
    mode: str = "extractive",
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Unified entry-point for answer generation.

    Dispatches to the appropriate backend based on ``mode``.

    Args:
        reduced_context: Compressed context string produced by ``SCRE.retrieve``.
        user_question: The original user question.
        mode: Answer mode.  Supported values:

            * ``"extractive"`` (default): Offline span extraction via spaCy.
            * ``"ollama"``: Generative answer from a local Ollama LLM.
        model_name: Ollama model name; only used when ``mode="ollama"``.
            Defaults to ``DEFAULT_MODEL_NAME``.

    Returns:
        A dict with keys:

        * ``"mode"`` (*str*): The mode that was used.
        * ``"answer"`` (*str*): The derived answer string.
    """
    if mode == "ollama":
        answer = answer_with_ollama(
            reduced_context=reduced_context,
            user_question=user_question,
            model_name=model_name,
        )
        return {
            "mode": "ollama",
            "answer": answer,
        }

    answer = extractive_answer(
        reduced_context=reduced_context,
        user_question=user_question,
    )
    return {
        "mode": "extractive",
        "answer": answer,
    }
