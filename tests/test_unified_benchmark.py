"""
Unit tests for tests/unified_benchmark.py's extract_json_workflows --
added this session to fix evaluate_dependency_recall's 0% score, which was
an artifact of re-analyzing SCRE's own JSON-rendered workflow output as if
it were prose (see the module docstring on extract_json_workflows for the
full root-cause explanation). No spaCy pipeline or embedder needed.
"""
import json

import csv

import pytest

from tests.unified_benchmark import extract_json_workflows, load_carried_strategies


def _workflow_json(name="Deploy", steps=None):
    steps = steps or ["1. Build", "2. Push", "3. Deploy"]
    return json.dumps({"type": "workflow", "name": name, "steps": steps}, indent=2)


def test_extract_json_workflows_finds_a_single_embedded_block():
    text = _workflow_json()
    found = extract_json_workflows(text)
    assert len(found) == 1
    assert found[0]["steps"] == ["1. Build", "2. Push", "3. Deploy"]


def test_extract_json_workflows_handles_mixed_prose_and_json():
    # Exactly the shape SCRE's build_compressed_context produces: plain
    # selected sentences newline-joined with an embedded workflow JSON
    # block for any preserved WorkflowUnit.
    text = "Decision: chose X.\n" + _workflow_json() + "\nReason: because Y."
    found = extract_json_workflows(text)
    assert len(found) == 1
    assert found[0]["name"] == "Deploy"


def test_extract_json_workflows_finds_multiple_blocks():
    text = _workflow_json("First") + "\n" + _workflow_json("Second", steps=["1. A", "2. B"])
    found = extract_json_workflows(text)
    assert {wf["name"] for wf in found} == {"First", "Second"}


def test_extract_json_workflows_ignores_non_workflow_json():
    text = json.dumps({"type": "not_a_workflow", "steps": ["1. A", "2. B"]})
    assert extract_json_workflows(text) == []


def test_extract_json_workflows_ignores_workflow_json_without_steps():
    text = json.dumps({"type": "workflow", "name": "Empty"})
    assert extract_json_workflows(text) == []


def test_extract_json_workflows_empty_when_no_json_present():
    assert extract_json_workflows("Just plain prose, no JSON here at all.") == []


def test_extract_json_workflows_ignores_malformed_json():
    text = '{"type": "workflow", "steps": [1, 2,'  # truncated/invalid
    assert extract_json_workflows(text) == []


def test_extract_json_workflows_handles_nested_braces_in_step_text():
    # A step string containing literal braces must not confuse the
    # brace-counting parser.
    text = json.dumps({"type": "workflow", "name": "X", "steps": ["1. Run f({x: 1})", "2. Done"]})
    found = extract_json_workflows(text)
    assert len(found) == 1
    assert found[0]["steps"][0] == "1. Run f({x: 1})"



# ---- carrying a saved strategy into a new run -------------------------------------------------
def _saved_run(tmp_path, questions=("q1", "q2")):
    jf, cf, hf = tmp_path / "r.json", tmp_path / "r.csv", tmp_path / "h.json"
    jf.write_text(json.dumps({"E_LLM": {"sps": 84.0}, "D_SCRE": {"sps": 83.0}}))
    with cf.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "question_id", "semantic_recall"])
        w.writeheader()
        w.writerows([{"strategy": s, "question_id": q, "semantic_recall": "1.0"} for s in ("E_LLM", "D_SCRE") for q in questions])
    hf.write_text(json.dumps([{"timestamp": "2026-09-14 20:52:40", "results": {"E_LLM": {"sps": 84.0}, "D_SCRE": {"sps": 83.0}}}]))
    return str(jf), str(cf), str(hf)


def test_carried_strategy_keeps_its_metrics_rows_and_origin(tmp_path):
    jf, cf, hf = _saved_run(tmp_path)
    results, rows = load_carried_strategies(["E_LLM"], ["q1", "q2"], jf, cf, hf)
    assert results["E_LLM"]["sps"] == 84.0 and results["E_LLM"]["carried_over_from"] == "2026-09-14 20:52:40"
    assert [r["strategy"] for r in rows] == ["E_LLM", "E_LLM"] and "D_SCRE" not in results


def test_carrying_over_refuses_a_run_that_covered_different_questions(tmp_path):
    jf, cf, hf = _saved_run(tmp_path)
    with pytest.raises(ValueError, match="does not cover exactly"):
        load_carried_strategies(["E_LLM"], ["q1", "q3"], jf, cf, hf)
    with pytest.raises(ValueError):
        load_carried_strategies(["F_missing"], ["q1", "q2"], jf, cf, hf)


def test_a_carried_row_stays_attributed_to_its_original_run(tmp_path):
    jf, cf, hf = _saved_run(tmp_path)
    first, _ = load_carried_strategies(["E_LLM"], ["q1", "q2"], jf, cf, hf)
    (tmp_path / "r.json").write_text(json.dumps({"E_LLM": first["E_LLM"]}))            # saved again after being carried
    again, _ = load_carried_strategies(["E_LLM"], ["q1", "q2"], jf, cf, hf)
    assert again["E_LLM"]["carried_over_from"] == "2026-09-14 20:52:40"
