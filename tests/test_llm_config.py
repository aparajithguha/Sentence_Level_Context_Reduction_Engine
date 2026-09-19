"""Offline tests for the harness LLM client (fake transport, no network)."""
import json

import pytest

from tests import llm_config as lc
from tests.llm_config import BudgetExceeded, DailyLimit, LLMClient, LLMConfig, LLMError, RoleConfig

MSG = [{"role": "user", "content": "hi"}]


def _cfg(models=("a", "b"), max_calls=10, attempts=1):
    role = RoleConfig(models=models, max_tokens=10, attempts_per_model=attempts)
    return LLMConfig(agent=role, judge=role, reducer=role, max_calls=max_calls)


def _client(script, cfg=None, ledger=None, monkeypatch=None):
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body["model"])
        return script(body["model"], len(calls))

    c = LLMClient(cfg or _cfg(), transport=transport, ledger=ledger, sleep=lambda s: None)
    c._key = "test-key"
    return c, calls


def _ok(text="hello"):
    return 200, {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 3}}


def test_first_model_answers():
    c, calls = _client(lambda m, n: _ok())
    r = c.chat("agent", MSG)
    assert (r.text, r.model, r.attempts) == ("hello", "a", 1) and calls == ["a"]


def test_falls_back_on_error_and_on_empty_reply():
    def script(model, n):
        if model == "a":
            return 429, {"error": {"message": "rate limited"}}
        return _ok()
    c, calls = _client(script)
    assert c.chat("agent", MSG).model == "b"

    c, _ = _client(lambda m, n: _ok("  ") if m == "a" else _ok("x"))
    assert c.chat("agent", MSG).model == "b"


def test_all_models_failing_raises_with_each_reason():
    c, _ = _client(lambda m, n: (503, {"error": {"message": "down"}}))
    with pytest.raises(LLMError, match="a -> 503:down.*b -> 503:down"):
        c.chat("judge", MSG)


def test_call_budget_is_enforced():
    c, _ = _client(lambda m, n: _ok(), cfg=_cfg(max_calls=1))
    c.chat("agent", MSG)
    with pytest.raises(BudgetExceeded):
        c.chat("agent", MSG)


def test_ledger_records_calls_but_never_the_key(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    c, _ = _client(lambda m, n: _ok(), ledger=ledger)
    c.chat("agent", MSG)
    raw = ledger.read_text()
    row = json.loads(raw.splitlines()[0])
    assert row["role"] == "agent" and row["ok"] and "test-key" not in raw


def test_env_override_and_defaults_use_different_families(monkeypatch):
    monkeypatch.setenv("SCRE_LLM_AGENT", "x/one:free, y/two:free")
    assert lc.default_config().agent.models == ("x/one:free", "y/two:free")
    monkeypatch.delenv("SCRE_LLM_AGENT")
    cfg = lc.default_config()
    family = lambda r: r.models[0].split("/")[0]
    assert len({family(cfg.agent), family(cfg.judge), family(cfg.reducer)}) == 3


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(lc, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        lc.load_api_key()


def test_daily_limit_stops_immediately_without_trying_other_models():
    c, calls = _client(lambda m, n: (429, {"error": {"message": "Rate limit exceeded: free-models-per-day."}}))
    with pytest.raises(DailyLimit):
        c.chat("agent", MSG)
    assert calls == ["a"]                      # no retry, no fallback
    assert issubclass(DailyLimit, LLMError)
