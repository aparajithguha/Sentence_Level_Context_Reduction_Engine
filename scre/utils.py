"""
utils.py
========
Pure, stateless helpers shared across the SCRE pipeline: a sentence-like
fallback object, entity extraction, text normalization, and IDF/Jaccard/
token-count math. No spaCy pipeline or embedder dependency of their own --
safe to import from anywhere in the package.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None


class DummySentence:
    """Fallback object mimicking a spaCy sentence span's ``.text`` attribute,
    used wherever a sentence-like object is needed but spaCy is unavailable
    or the text isn't backed by a real parse (e.g. for IDF computation)."""
    def __init__(self, text_val: str):
        self.text = text_val


def extract_entities(sent: Any) -> set[str]:
    """Extract lowercased entity/proper-noun strings from a parsed sentence span.

    Reuses the spaCy parse already performed during segmentation rather than
    re-parsing -- NER alone can mislabel domain-specific proper nouns (a
    small model may tag "PostgreSQL" as GPE or "Redis" as PERSON), but it
    still reliably finds *that* they are entities; PROPN tokens are unioned
    in as a fallback net for anything NER misses entirely.

    Args:
        sent: A sentence-like object. Returns an empty set unless it exposes
            spaCy's ``.ents`` (i.e. is a real parsed ``Span``/``Doc``, not a
            ``DummySentence`` fallback used in regex-only mode).

    Returns:
        A set of lowercased, stripped entity/proper-noun strings.
    """
    if not hasattr(sent, "ents"):
        return set()
    entities = {ent.text.lower().strip() for ent in sent.ents}
    entities |= {tok.text.lower().strip() for tok in sent if getattr(tok, "pos_", None) == "PROPN"}
    return {e for e in entities if e}


def normalize_text(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space and strip leading/trailing whitespace.

    Args:
        text: The raw input string.

    Returns:
        A normalised string with no leading/trailing whitespace and no
        internal runs of multiple whitespace characters.
    """
    return re.sub(r"\s+", " ", text).strip()


def compute_idf(sentences: list) -> dict[str, float]:
    """Compute Inverse Document Frequency (IDF) weights for all terms across a list of sentences.

    IDF is used during scoring to upweight query terms that are rare across
    the document corpus, reducing the noise contribution of ubiquitous words
    that survive stop-word filtering.

    The formula used is the smoothed variant::

        idf(t) = log(N / (1 + df(t))) + 1

    where *N* is the total number of sentences and *df(t)* is the number of
    sentences containing term *t*.

    Args:
        sentences: A list of sentence-like objects that expose a ``.text``
            attribute (e.g., spaCy ``Span`` objects or ``DummySentence``
            instances used as a fallback).

    Returns:
        A dictionary mapping each term (lowercased, length ≥ 2) to its IDF
        weight.  Returns an empty dict if ``sentences`` is empty.
    """
    doc_freq = Counter()
    total_docs = len(sentences)

    if total_docs == 0:
        return {}

    for sent in sentences:
        # Get unique terms in this sentence
        words = set(re.sub(r"\W+", " ", sent.text.lower()).split())
        doc_freq.update(words)

    # Calculate IDF: log(total_docs / doc_freq)
    idf = {}
    for term, freq in doc_freq.items():
        if term and len(term) >= 2:  # Skip single chars and empty
            idf[term] = math.log(total_docs / (1 + freq)) + 1

    return idf


def jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    a_terms = set(re.findall(r"\w+", a.lower()))
    b_terms = set(re.findall(r"\w+", b.lower()))

    if not a_terms or not b_terms:
        return 0.0

    return len(a_terms & b_terms) / len(a_terms | b_terms)


def estimate_tokens(text: str) -> int:
    """Accurate token estimation using tiktoken, fallback to char heuristic."""
    if not text.strip():
        return 0
    if tiktoken is not None:
        # Use cl100k_base which is standard for GPT-4/latest LLMs
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, math.ceil(len(text) / 4))
