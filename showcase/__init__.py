"""
showcase
========
A demo web UI for the ``scre`` library. This package is a *consumer* of
``scre`` -- it imports the library's public API (``SCRE`` plus its
public submodules) and never edits anything under ``scre/``. Kept
separate on purpose: ``scre`` is meant to be published as a standalone
library, and a UI-specific need (showing per-sentence scoring detail
that ``SCRE.reduce()`` intentionally doesn't return) belongs in the
consumer, not bolted onto the library's API.
"""
