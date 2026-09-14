"""
server.py
=========
Small Flask app that serves the demo UI (``showcase/static/index.html``)
and one JSON endpoint backing it. This file, and everything else under
``showcase/``, is a *consumer* of the ``scre`` library -- it never edits
``scre/``.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from showcase.reducer_adapter import ExplainedReducer, MODEL_NAMES

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="")

# One SCRE engine, one adapter, shared across requests -- SCRE itself is
# documented as stateless per-call, so reusing one instance (rather than
# constructing a fresh one per request) is exactly the "long-running
# process reuses one engine instance" pattern the library is designed for.
_reducer = ExplainedReducer()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/reduce")
def reduce_endpoint():
    payload = request.get_json(silent=True) or {}
    content = (payload.get("content") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    result = _reducer.reduce_with_explanation(content)
    return jsonify(result)


@app.post("/api/understand")
def understand_endpoint():
    payload = request.get_json(silent=True) or {}
    original = (payload.get("original") or "").strip()
    reduced = (payload.get("reduced") or "").strip()
    model_key = payload.get("model", "qwen3")

    if not original or not reduced:
        return jsonify({"error": "original and reduced are both required"}), 400
    if model_key not in MODEL_NAMES:
        return jsonify({"error": f"unknown model '{model_key}'"}), 400

    try:
        result = _reducer.compare_understanding(original, reduced, model_key)
    except Exception as exc:  # Ollama not running, model not pulled, etc.
        return jsonify({"error": str(exc)}), 502
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5050, debug=True)
