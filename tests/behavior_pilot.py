"""
behavior_pilot.py
=================
Round-1 pilot: does an agent behave the same on a reduced system prompt?

For each case (system prompt, task, automatic checks) it builds one prompt per
approach, runs the agent on each, and scores the checks. Only this harness
calls an LLM; ``scre`` never does.

  orig1, orig2   original prompt, run twice -> the agent's own run-to-run noise
  ctrl_deleted   original with the section holding the key rule deleted -> checks must fail
  v1             redundancy-only reduction (examples/duplicates)
  random         random sections dropped, same size as A_strict
  A              deterministic task matcher
  A_strict       same, stricter matching (fewer sections match the task)
  B              LLM chooses which sections to drop (ids only; code rebuilds verbatim)
  C              LLM tightens wording block by block; each rewrite is verified by code
  D              LLM rewrites the whole prompt freely (reference for ceiling and risk)

Run: python -m tests.behavior_pilot [case_id ...]     (results in prompt_datasets/results/)
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scre.blocks import parse_blocks, rebuild
from scre.config import PromptConfig
from scre.prompt_modules import find_modules
from scre.prompt_reducer import PromptReducer
from scre.roles import classify
from tests.llm_config import BudgetExceeded, LLMClient, LLMError

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "System-Prompts-main"
OUT = ROOT / "prompt_datasets" / "results" / "behavior_round1"


# --------------------------------------------------------------------------- cases
def _norm(r: str) -> str:
    return r.replace("**_", "***")  # Codex.md's own corrupted markers


def codex_begin_patch(r: str) -> bool:
    return re.search(r"\*\*\*\s*Begin Patch", _norm(r), re.I) is not None


def codex_add_file(r: str) -> bool:
    n = _norm(r)
    return (re.search(r"\*\*\*\s*Add File:\s*hello\.txt", n, re.I) is not None
            and re.search(r"^\+\s*hi\s*$", n, re.M) is not None)


_BOLT_BLOCK = re.compile(r"<boltArtifact\b.*?</boltArtifact>", re.S)
_BOLT_FILES = re.compile(r'<boltAction\b[^>]*filePath="([^"]+)"[^>]*>(.*?)</boltAction>', re.S)


def bolt_format(r: str) -> bool:
    return "<boltArtifact" in r and any(p == "package.json" for p, _ in _BOLT_FILES.findall(r))


def bolt_package_first(r: str) -> bool:
    paths = [p for p, _ in _BOLT_FILES.findall(r)]
    return len(paths) >= 2 and paths[0] == "package.json"


def bolt_no_artifact_word(r: str) -> bool:
    return "artifact" not in _BOLT_BLOCK.sub("", r).lower()


def bolt_two_space(r: str) -> bool:
    code = "\n".join(c for p, c in _BOLT_FILES.findall(r) if p.endswith((".js", ".mjs", ".cjs")))
    widths = [len(l) - len(l.lstrip(" ")) for l in code.splitlines() if l.startswith(" ")]
    return bool(widths) and "\t" not in code and min(widths) == 2 and all(w % 2 == 0 for w in widths)


@dataclass
class Case:
    id: str
    file: str
    task: str
    checks: dict[str, Callable[[str], bool]]
    key_module: str                    # name of the section deleted in the control arm
    note: str = ""


CASES = [
    Case("codex_hello", "Codex.md",
         "Create a file named hello.txt in the current directory containing the single line: hi",
         {"begin_patch": codex_begin_patch, "add_file_hello": codex_add_file}, "applypatch"),
    Case("bolt_chalk", "Bolt.new.md",
         "Create a Node.js project where index.js defines a function greet(name) that returns a greeting, "
         "and prints greet('world') in green using the chalk package.",
         {"artifact_format": bolt_format, "package_json_first": bolt_package_first,
          "no_word_artifact": bolt_no_artifact_word, "two_space_indent": bolt_two_space}, "artifact_info"),
]


# --------------------------------------------------------------------------- arms
def delete_module(text: str, name: str) -> str:
    blocks = parse_blocks(text)
    classify(blocks)
    mods = [m for m in find_modules(blocks, PromptConfig()) if name in m.name]
    if not mods:
        raise ValueError(f"no module named like {name!r}")
    gone = set(mods[0].block_ids)
    return rebuild(blocks, {b.id for b in blocks if b.id not in gone})


def random_drop(text: str, target_chars: int, seed: int = 7) -> str:
    red = PromptReducer()
    mods = [m for m in red.outline(text) if m["class"] != "global" and m["chars"] >= PromptConfig().module_min_chars]
    random.Random(seed).shuffle(mods)
    chosen: list[int] = []
    out = red.reduce(text)["context"]
    for m in mods:
        if len(out) <= target_chars:
            break
        chosen.append(m["id"])
        out = red.reduce(text, drop_modules=chosen)["context"]
    return out


_JSON = re.compile(r"\{.*\}", re.S)


def llm_select(text: str, task: str, client: LLMClient) -> tuple[str, dict]:
    red = PromptReducer()
    outline = red.outline(text)
    lines = [f"[{m['id']}] {m['name']} | {m['chars']} chars | {m['class']} | {m['preview'][:200]}" for m in outline]
    system = (
        "You help shrink an AI agent's system prompt for ONE task. You get the task and the prompt's sections "
        "(id, title, size, class, short preview). Pick the sections the agent can do without for this task. "
        "Think about what the agent must do to finish the task: which tools it must call, the exact formats it "
        "must produce, and any safety or style rules that apply on the way. Drop a section only if you are "
        "confident it cannot matter; when unsure, keep it. Reply with JSON only: "
        '{"drop": [ids], "why": {"id": "short reason"}}'
    )
    reply = client.chat("reducer", [{"role": "system", "content": system},
                                    {"role": "user", "content": f"TASK: {task}\n\nSECTIONS:\n" + "\n".join(lines)}],
                        max_tokens=3000)
    m = _JSON.search(reply.text)
    drop: list[int] = []
    try:
        drop = [int(i) for i in json.loads(m.group(0)).get("drop", [])] if m else []
    except (ValueError, TypeError, AttributeError):
        drop = []
    valid = {x["id"] for x in outline}
    drop = [i for i in drop if i in valid]
    r = red.reduce(text, task=task, drop_modules=drop)
    kept_by_guard = [x["name"] for x in r["metadata"]["modules"] if x["id"] in drop and x["kept"]]
    return r["context"], {"asked_to_drop": [x["name"] for x in outline if x["id"] in drop], "kept_by_guard": kept_by_guard,
                          "reducer_model": reply.model, "reducer_secs": reply.secs}


_STRONG = ["must", "never", "always", "do not", "don't", "should not", "shouldn't", "required", "forbidden",
           "prohibited", "unless", "until", "at most", "at least", "only", "before", "after"]
_PROSE = {"rules", "procedure", "text", "reasoning", "reference", "format", "tool_def"}


def _atoms(s: str) -> set[str]:
    return (set(re.findall(r"\d+(?:\.\d+)?", s)) | set(re.findall(r"`[^`]+`", s)) | set(re.findall(r'"[^"\n]{1,60}"', s))
            | set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", s)) | set(re.findall(r"</?[A-Za-z_][\w\-]*>", s)))


def verify_rewrite(old: str, new: str) -> tuple[bool, str]:
    if not new.strip():
        return False, "empty"
    if len(new.strip()) >= len(old.strip()):
        return False, "not shorter"
    mo, mn = re.match(r"\s*([-*+]|\d+[.)])", old), re.match(r"\s*([-*+]|\d+[.)])", new)
    if (mo.group(1) if mo else None) != (mn.group(1) if mn else None):
        return False, "list marker changed"
    lost = _atoms(old) - _atoms(new)
    if lost:
        return False, f"lost {sorted(lost)[:2]}"
    for w in _STRONG:
        if len(re.findall(rf"\b{re.escape(w)}\b", new.lower())) < len(re.findall(rf"\b{re.escape(w)}\b", old.lower())):
            return False, f"fewer '{w}'"
    return True, "ok"


def tighten_rewrites(text: str, client: LLMClient) -> tuple[dict[int, str], dict]:
    """One LLM call per prompt (task-independent): verified rewrites keyed by block id."""
    base = PromptReducer().reduce(text)
    blocks = [b for b in base["blocks"] if b["kept"] and b["kind"] in ("paragraph", "item") and b["role"] in _PROSE and b["chars"] >= 120]
    system = (
        "You tighten the wording of an AI agent's system prompt without changing what it requires. You get numbered "
        "text blocks. Return every block again, shortened, as a line '### <id>' followed by the rewritten text. "
        "Keep every instruction, condition, number, name, quoted string, code identifier and tag. Keep words like "
        "must/never/always/only/unless/before/after. Keep any list marker at the start. Do not merge, reorder or "
        "drop blocks. Remove only filler, repetition and needless words. If a block cannot be shortened safely, "
        "return it unchanged."
    )
    user = "\n".join(f"### {b['id']}\n{b['text'].strip()}" for b in blocks)
    reply = client.chat("reducer", [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=9000)
    parts = re.split(r"(?m)^### (\d+)\s*$", reply.text)
    proposed = {int(parts[i]): parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}
    by_id = {b["id"]: b for b in base["blocks"]}
    accepted: dict[int, str] = {}
    reasons: dict[str, int] = {}
    for i, new in proposed.items():
        if i not in by_id:
            continue
        ok, why = verify_rewrite(by_id[i]["text"], new)
        if ok:
            accepted[i] = new
        else:
            reasons[why.split(" ")[0]] = reasons.get(why.split(" ")[0], 0) + 1
    return accepted, {"blocks_sent": len(blocks), "rewritten_ok": len(accepted), "rejected": sum(reasons.values()),
                      "reject_reasons": reasons, "reducer_model": reply.model, "reducer_secs": reply.secs}


def apply_rewrites(blocks: list[dict], rewrites: dict[int, str]) -> str:
    """Join the kept blocks, swapping in verified rewrites (keeps each block's own indent and trailing space)."""
    out = []
    for b in blocks:
        if not b["kept"]:
            continue
        t = b["text"]
        if b["id"] in rewrites:
            t = re.match(r"\s*", t).group(0) + rewrites[b["id"]].strip() + t[len(t.rstrip()):]
        out.append(t)
    return "".join(out).rstrip()


def llm_tighten(text: str, client: LLMClient) -> tuple[str, dict]:
    rewrites, info = tighten_rewrites(text, client)
    return apply_rewrites(PromptReducer().reduce(text)["blocks"], rewrites), info


def llm_rewrite(text: str, client: LLMClient) -> tuple[str, dict]:
    system = (
        "Rewrite this system prompt to be much shorter, so that an AI agent following the rewritten prompt would "
        "behave exactly as it would with the original. Keep every rule, required format, tool-usage detail and "
        "constraint that can change behavior. Output only the rewritten prompt."
    )
    reply = client.chat("reducer", [{"role": "system", "content": system}, {"role": "user", "content": text}], max_tokens=9000)
    out = re.sub(r"\A```\w*\n|\n```\s*\Z", "", reply.text.strip())
    return out, {"reducer_model": reply.model, "reducer_secs": reply.secs}


# --------------------------------------------------------------------------- run
@dataclass
class ArmResult:
    arm: str
    chars: int
    saved_pct: float
    passed: dict[str, bool] = field(default_factory=dict)
    model: str = ""
    secs: float = 0.0
    info: dict = field(default_factory=dict)
    response: str = ""
    error: str = ""


def run_case(case: Case, client: LLMClient, arms_wanted: list[str] | None = None) -> list[ArmResult]:
    text = (PROMPTS / case.file).read_text(encoding="utf-8")
    red, strict = PromptReducer(), PromptReducer(PromptConfig(module_max_df_ratio=0.25))
    prompts: dict[str, tuple[str, dict]] = {"orig1": (text, {}), "orig2": (text, {}),
                                            "ctrl_deleted": (delete_module(text, case.key_module), {}),
                                            "v1": (red.reduce(text)["context"], {})}
    a_strict = strict.reduce(text, task=case.task)
    prompts["A"] = (red.reduce(text, task=case.task)["context"], {})
    prompts["A_strict"] = (a_strict["context"], {"dropped": [m["name"] for m in a_strict["metadata"]["modules"] if not m["kept"]]})
    prompts["random"] = (random_drop(text, len(a_strict["context"])), {})
    results: list[ArmResult] = []
    outdir = OUT / case.id
    outdir.mkdir(parents=True, exist_ok=True)

    llm_arms = {"B": lambda: llm_select(text, case.task, client), "C": lambda: llm_tighten(text, client),
                "D": lambda: llm_rewrite(text, client)}
    order = ["orig1", "orig2", "ctrl_deleted", "v1", "random", "A", "A_strict", "B", "C", "D"]
    for arm in order:
        if arms_wanted and arm not in arms_wanted:
            continue
        try:
            prompt, info = prompts[arm] if arm in prompts else llm_arms[arm]()
        except (LLMError, ValueError) as e:
            results.append(ArmResult(arm, 0, 0.0, error=f"reducer failed: {e}"))
            print(f"  {arm:13} reducer failed: {str(e)[:100]}", flush=True)
            continue
        (outdir / f"{arm}.txt").write_text(prompt, encoding="utf-8")
        res = ArmResult(arm, len(prompt), round(100 * (1 - len(prompt) / len(text)), 1), info=info)
        try:
            reply = client.chat("agent", [{"role": "system", "content": prompt}, {"role": "user", "content": case.task}],
                                models=(client.config.agent.models[0],), attempts=4)
            res.response, res.model, res.secs = reply.text, reply.model, reply.secs
            res.passed = {name: bool(fn(reply.text)) for name, fn in case.checks.items()}
        except LLMError as e:
            res.error = f"agent failed: {str(e)[:120]}"
        results.append(res)
        ok = sum(res.passed.values())
        print(f"  {arm:13} {res.chars:6} chars  saved {res.saved_pct:5.1f}%  checks {ok}/{len(case.checks)}  {res.error}", flush=True)
    return results


def report(case: Case, results: list[ArmResult]) -> None:
    print(f"\n=== {case.id}  ({case.file}, task: {case.task[:70]}...)")
    print(f"{'arm':13} {'chars':>6} {'saved':>6}  " + "  ".join(f"{c[:14]:14}" for c in case.checks) + "  verdict")
    base = {r.arm: r for r in results}
    o1 = base.get("orig1")
    for r in results:
        cells = "  ".join(f"{('PASS' if r.passed.get(c) else 'FAIL') if r.passed else '--':14}" for c in case.checks)
        verdict = ""
        if r.error:
            verdict = r.error[:40]
        elif o1 and o1.passed and r.arm not in ("orig1",):
            same = r.passed == o1.passed
            verdict = "same as original" if same else "DIFFERS from original"
        print(f"{r.arm:13} {r.chars:6} {r.saved_pct:5.1f}%  {cells}  {verdict}")


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = LLMClient()
    wanted = [c for c in CASES if not argv or c.id in argv]
    allres = {}
    try:
        for case in wanted:
            print(f"\n>>> {case.id}", flush=True)
            res = run_case(case, client)
            allres[case.id] = res
            report(case, res)
    except BudgetExceeded as e:
        print("STOPPED:", e)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"results_{stamp}.json").write_text(json.dumps(
        {cid: [r.__dict__ for r in rs] for cid, rs in allres.items()}, indent=1, default=str), encoding="utf-8")
    print(f"\nLLM calls this run: {client.calls}  (ledger: prompt_datasets/results/llm_calls.jsonl)")


if __name__ == "__main__":
    main(sys.argv[1:])
