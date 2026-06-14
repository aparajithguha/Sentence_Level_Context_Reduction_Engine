from __future__ import annotations

import re
from typing import Any

import spacy


NLP = spacy.load("en_core_web_sm")

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency at runtime
    ollama = None


DEFAULT_MODEL_NAME = "gemma4:e2b"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_terms(text: str) -> list[str]:
    doc = NLP(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
    ]


def score_line_for_question(line: str, user_question: str) -> float:
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
    return f"""Answer the user's question using only the provided context.

Note: The context may contain structured JSON objects representing procedural workflows or lists. Treat these JSON blocks as valid factual evidence.

If the context does not contain the answer, reply exactly with:
I cannot determine that from the context.

Context:
{reduced_context}

Question:
{user_question}
"""


def extractive_answer(
    reduced_context: str,
    user_question: str,
) -> str:
    """
    Lightweight offline fallback for demo and evaluation.
    """

    context_lines = [
        re.sub(r"^\[S\d+\]\s*", "", line).strip()
        for line in reduced_context.splitlines()
        if line.strip()
    ]

    if not context_lines:
        return "I cannot determine that from the context."

    question_doc = NLP(user_question)
    wh_word = next(
        (
            token.lower_
            for token in question_doc
            if token.lower_ in {"who", "what", "where", "when", "why"}
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

    if wh_word == "why":
        for line in ranked_lines:
            lowered = line.lower()
            if "because" in lowered:
                start = lowered.index("because") + len("because")
                causal_clause = line[start:].strip().rstrip(".")
                return causal_clause.split(",", 1)[0].strip()
        return ranked_lines[0]

    return ranked_lines[0]


def answer_with_ollama(
    reduced_context: str,
    user_question: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
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


def answer_question(
    reduced_context: str,
    user_question: str,
    mode: str = "extractive",
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
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
