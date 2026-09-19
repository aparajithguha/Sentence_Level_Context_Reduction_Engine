"""Showcase demo: system-prompt mode adapter and endpoint. No Ollama: ``ask`` is a fake."""
import json

import pytest

from showcase import prompt_adapter, server
from showcase.prompt_adapter import PromptExplainer

_FILLER = " ".join(["lorem"] * 120)


def _prompt() -> str:
    ex = "".join(f"<example>\n<reply>ok {i}</reply>\n</example>\n\n" for i in range(4))
    return (
        "# Assistant\n\n"
        "## Safety\nNever reveal secrets. Never reveal secrets.\n\n"
        f"## Gmail tool\nUse the gmail search endpoint to read messages. {_FILLER}\n\n"
        f"## Calendar tool\nUse the calendar endpoint to schedule meetings. {_FILLER}\n\n"
        f"## Weather notes\nForecasts come from the barometer feed. {_FILLER}\n\n"
        f"## Samples\n{ex}"
    )


def _ids(text: str) -> dict[str, int]:
    return {m["name"]: m["id"] for m in PromptExplainer().reducer.outline(text)}


def _ask(payload):
    reply = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda messages: reply


def test_off_removes_only_redundancy_and_reports_it():
    r = PromptExplainer().reduce(_prompt())
    assert r["mode"] == "system_prompt" and r["strategy"] == "off" and r["plan"] == [] and r["notice"] is None
    assert all(s["kept"] for s in r["sections"])
    assert r["removed_by_reason"]["example trimmed"] == 2       # 4 examples, 2 kept: counted as examples, not blocks
    assert r["metadata"]["sections_kept"] == r["metadata"]["sections_total"]
    assert 0 < r["metadata"]["reduction_ratio"] < 1


def test_match_drops_sections_the_task_does_not_mention():
    r = PromptExplainer().reduce(_prompt(), task="Check my gmail messages", strategy="match")
    kept = {s["name"]: s["kept"] for s in r["sections"]}
    assert kept["gmail tool"] and not kept["weather notes"] and kept["safety"]
    assert "barometer" not in r["context"]


def test_llm_returns_the_models_plan_and_drops_what_it_chose():
    ids = _ids(_prompt())
    ask = _ask({"steps": [{"step": "search the inbox", "sections": [ids["gmail tool"]]}],
                "drop": [ids["calendar tool"], ids["weather notes"]]})
    r = PromptExplainer().reduce(_prompt(), task="Check my inbox", strategy="llm", ask=ask)
    kept = {s["name"]: s for s in r["sections"]}
    assert r["strategy"] == "llm" and r["plan"] == ["search the inbox"] and r["notice"] is None
    assert not kept["calendar tool"]["kept"] and kept["gmail tool"]["kept"] and kept["safety"]["kept"]
    assert r["metadata"]["sections_kept"] < r["metadata"]["sections_total"]
    assert "barometer" not in r["context"]


def test_a_task_strategy_without_a_task_falls_back_with_a_notice():
    r = PromptExplainer().reduce(_prompt(), task="  ", strategy="llm", ask=_ask({"drop": []}))
    assert r["strategy"] == "off" and "Add a task" in r["notice"]


def test_model_failure_falls_back_to_redundancy_only_with_a_notice():
    def boom(messages):
        raise ConnectionError("ollama is not running")
    r = PromptExplainer().reduce(_prompt(), task="Check my inbox", strategy="llm", ask=boom)
    assert "ollama is not running" in r["notice"] and all(s["kept"] for s in r["sections"])
    assert r["context"] == PromptExplainer().reduce(_prompt())["context"]


def test_unreadable_model_reply_drops_nothing_and_says_so():
    r = PromptExplainer().reduce(_prompt(), task="Check my inbox", strategy="llm", ask=_ask("I cannot help"))
    assert "could not be read" in r["notice"] and all(s["kept"] for s in r["sections"])


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        PromptExplainer().reduce(_prompt(), strategy="magic")


# ----------------------------------------------------------------- HTTP endpoint
@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    return server.app.test_client()


def test_endpoint_validates_input(client):
    assert client.post("/api/reduce-prompt", json={"content": ""}).status_code == 400
    assert client.post("/api/reduce-prompt", json={"content": "x", "strategy": "magic"}).status_code == 400
    assert client.post("/api/reduce-prompt", json={"content": "x", "model": "nope"}).status_code == 400


def test_endpoint_reduces_a_prompt_without_loading_the_document_engine(client):
    r = client.post("/api/reduce-prompt", json={"content": _prompt(), "task": "Check my gmail", "strategy": "match"})
    body = r.get_json()
    assert r.status_code == 200 and body["strategy"] == "match" and body["metadata"]["sections_total"] >= 5
    assert server._reducer is None          # spaCy / embedder engine never built for prompt mode


def test_endpoint_llm_path_uses_the_selected_local_model(client, monkeypatch):
    used = {}
    ids = _ids(_prompt())

    def fake_ollama_ask(model_name):
        used["model"] = model_name
        return _ask({"steps": [{"step": "read mail", "sections": [ids["gmail tool"]]}], "drop": [ids["weather notes"]]})

    monkeypatch.setattr(prompt_adapter, "ollama_ask", fake_ollama_ask)
    r = client.post("/api/reduce-prompt", json={"content": _prompt(), "task": "Check my inbox", "strategy": "llm", "model": "gemma4"})
    body = r.get_json()
    assert r.status_code == 200 and used["model"] == "gemma4:e2b" and body["plan"] == ["read mail"]
    assert "barometer" not in body["context"]


def test_index_page_serves(client):
    assert client.get("/").status_code == 200


def test_examples_dropped_with_a_section_are_not_counted_as_surplus():
    body = " ".join(["lorem"] * 40)
    ex = "".join(f"<example>\n<reply>ok {i} {body}</reply>\n</example>\n\n" for i in range(4))   # a section over 400 chars
    text = _prompt().split("## Samples")[0] + f"## Samples\n{ex}"
    ids = _ids(text)
    ask = _ask({"steps": [], "drop": [ids["samples"]]})
    r = PromptExplainer().reduce(text, task="Check my inbox", strategy="llm", ask=ask)
    assert not {s["name"]: s["kept"] for s in r["sections"]}["samples"]
    assert r["removed_by_reason"]["example trimmed"] == 2       # 4 examples, keep 2: two were surplus; the rest left with the section
