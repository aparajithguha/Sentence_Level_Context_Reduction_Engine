"""
models.py
=========
Process-wide singleton loaders for the ML models SCRE depends on.

SCRE is meant to run inline on a live prompt path — a fresh reduction per
request. That requires per-request *data* (units, graphs) but a shared,
loaded-once *model* (spaCy pipeline, sentence-transformer). Loading either
model inside every request-scoped object construction turns each request
into a multi-second model load. These functions load once per process and
hand back the same cached instance on every subsequent call.
"""
from __future__ import annotations

import threading
from typing import Any

try:
    import spacy
except ImportError:
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


_lock = threading.Lock()
_nlp_cache: dict[str, Any] = {}
_embedder_cache: dict[str, Any] = {}


def get_nlp(model: str = "en_core_web_sm"):
    """Return a cached spaCy pipeline for ``model``, loading it once.

    Args:
        model: spaCy model name. Defaults to ``"en_core_web_sm"``.

    Returns:
        The loaded spaCy ``Language`` object, or ``None`` if spaCy is not
        installed.
    """
    if spacy is None:
        return None
    if model not in _nlp_cache:
        with _lock:
            if model not in _nlp_cache:
                _nlp_cache[model] = spacy.load(model)
    return _nlp_cache[model]


def get_embedder(model: str = "all-MiniLM-L6-v2"):
    """Return a cached ``SentenceTransformer`` for ``model``, loading it once.

    Args:
        model: Sentence-transformer model name. Defaults to
            ``"all-MiniLM-L6-v2"``.

    Returns:
        The loaded ``SentenceTransformer`` object, or ``None`` if
        sentence-transformers is not installed.
    """
    if SentenceTransformer is None:
        return None
    if model not in _embedder_cache:
        with _lock:
            if model not in _embedder_cache:
                _embedder_cache[model] = SentenceTransformer(model)
    return _embedder_cache[model]
