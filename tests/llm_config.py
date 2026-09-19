"""
llm_config.py
=============
Test-harness-only OpenRouter config and client. The ``scre`` library never
imports this: the reduction engine stays LLM-free, and an optional LLM
reducer is plugged in by passing a callable.

Roles (families differ on purpose, so the judge is not grading its own kind):
  agent    -- runs the original / reduced system prompt on a task
  judge    -- scores agent behavior when a deterministic check can't
  reducer  -- optional LLM reduction arms (pick block ids / tighten wording)

Each role has an ordered model list: the first that answers wins, the rest
are fallbacks (free-tier models return 429/5xx or empty replies often).
Override with SCRE_LLM_AGENT / SCRE_LLM_JUDGE / SCRE_LLM_REDUCER
(comma-separated model ids). SCRE_LLM_MAX_CALLS caps calls per process.

The API key comes from OPENROUTER_API_KEY (environment, else the git-ignored
.env). It is never logged or written to the ledger.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LEDGER = ROOT / "prompt_datasets" / "results" / "llm_calls.jsonl"  # git-ignored

LING = "inclusionai/ling-3.0-flash-vl:free"
DEEPSEEK = "deepseek/deepseek-v4-flash-0731:free"
NEMOTRON_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b:free"
NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b:free"


@dataclass(frozen=True)
class RoleConfig:
    models: tuple[str, ...]
    max_tokens: int
    temperature: float = 0.0
    timeout_s: int = 120
    attempts_per_model: int = 2


@dataclass(frozen=True)
class LLMConfig:
    agent: RoleConfig
    judge: RoleConfig
    reducer: RoleConfig
    max_calls: int = 80


def _models(env: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env, "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or default


def default_config() -> LLMConfig:
    return LLMConfig(
        agent=RoleConfig(_models("SCRE_LLM_AGENT", (LING, DEEPSEEK)), max_tokens=1500),
        judge=RoleConfig(_models("SCRE_LLM_JUDGE", (DEEPSEEK, NEMOTRON_SUPER)), max_tokens=800),
        # Nemotron-ultra once looped to the token cap, so the cap and timeout stay firm.
        reducer=RoleConfig(_models("SCRE_LLM_REDUCER", (NEMOTRON_ULTRA, DEEPSEEK)),
                           max_tokens=6000, timeout_s=180),
        max_calls=int(os.environ.get("SCRE_LLM_MAX_CALLS", "80")),
    )


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "OPENROUTER_API_KEY":
                    key = value.strip().strip("'\"")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (environment or .env)")
    return key


@dataclass
class Reply:
    text: str
    model: str
    secs: float
    attempts: int
    usage: dict


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class DailyLimit(LLMError):
    """The free-tier daily request cap is used up; retrying or switching models cannot help."""


def _http_transport(url: str, headers: dict, body: dict, timeout: int) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"error": {"message": str(e)}}


class LLMClient:
    """Calls a role's models in order until one returns non-empty text."""

    def __init__(self, config: LLMConfig | None = None, transport=None, ledger: Path | None = LEDGER,
                 sleep=time.sleep):
        self.config = config or default_config()
        self._transport = transport or _http_transport
        self._ledger = ledger
        self._sleep = sleep
        self.calls = 0
        self._key: str | None = None

    def _headers(self) -> dict:
        if self._key is None:
            self._key = load_api_key()
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _log(self, row: dict) -> None:
        if self._ledger is None:
            return
        self._ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def chat(self, role: str, messages: list[dict], **overrides) -> Reply:
        cfg: RoleConfig = getattr(self.config, role)
        max_tokens = overrides.get("max_tokens", cfg.max_tokens)
        temperature = overrides.get("temperature", cfg.temperature)
        models = tuple(overrides.get("models", cfg.models))       # e.g. (primary,) to forbid fallback
        per_model = overrides.get("attempts", cfg.attempts_per_model)
        attempts = 0
        errors: list[str] = []
        for model in models:
            for n in range(per_model):
                if self.calls >= self.config.max_calls:
                    raise BudgetExceeded(f"call budget of {self.config.max_calls} reached")
                self.calls += 1
                attempts += 1
                t0 = time.time()
                status, data = self._transport(
                    ENDPOINT, self._headers(),
                    {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                    cfg.timeout_s,
                )
                secs = round(time.time() - t0, 2)
                text = ""
                if status == 200 and "error" not in data:
                    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                ok = bool(text.strip())
                err = "" if ok else f"{status}:{(data.get('error') or {}).get('message', 'empty reply')}"[:160]
                self._log({"ts": round(t0), "role": role, "model": model, "status": status, "secs": secs,
                           "ok": ok, "error": err, "usage": data.get("usage") or {}})
                if ok:
                    return Reply(text, model, secs, attempts, data.get("usage") or {})
                if status == 429 and "per-day" in err:
                    raise DailyLimit("OpenRouter free-models-per-day cap reached; resume after the reset "
                                     "(midnight UTC) or add credits")
                errors.append(f"{model} -> {err}")
                if n + 1 < per_model:
                    self._sleep(min(30, 2 ** (n + 1)) + random.random())
        raise LLMError(f"all models failed for role '{role}': " + "; ".join(errors))


if __name__ == "__main__":  # one tiny call per role, to check the wiring
    client = LLMClient()
    for role in ("agent", "judge", "reducer"):
        try:
            r = client.chat(role, [{"role": "user", "content": "Reply with the single word OK."}], max_tokens=200)
            print(f"{role:8} {r.model:48} {r.secs:>5}s  attempts={r.attempts}  reply={r.text.strip()[:30]!r}")
        except (LLMError, BudgetExceeded) as e:
            print(f"{role:8} FAILED: {e}")
