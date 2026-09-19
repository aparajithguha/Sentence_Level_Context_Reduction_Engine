"""
behavior_round4.py
==================
Round 4: a fresh test of B2 (and A) on prompts and tasks that played no part in building either.

Prompts: ChatGPT 5.md and V0.md (neither was used in rounds 1-3).
Cases and checks were written before any run. Rules fixed in advance:
  * a case counts only if the original passes all its checks AND the deleted-section control fails at least one;
  * an approach that differs from the original is re-run once (with the original) to tell drift from noise.

Arms:
  orig1          original prompt
  ctrl_deleted   original minus the section holding the key rule
  A              deterministic task matcher (+ cross-reference guards), no LLM
  B2             LLM picks sections from their full text after listing the agent's steps (+ guards)

Run: python -m tests.behavior_round4 [case_id ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scre.prompt_reducer import PromptReducer
from tests.behavior_pilot import PROMPTS, Case, delete_module
from tests.behavior_round3 import select_b2
from tests.llm_config import BudgetExceeded, DailyLimit, LLMClient, LLMError

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "prompt_datasets" / "results" / "behavior_round4"


# ----------------------------------------------------------------- checks (written before any run)
def has(*needles: str):
    return lambda r: all(n.lower() in r.lower() for n in needles)


def gpt5_rrule(r: str) -> bool:
    return all(s in r for s in ("RRULE", "BYDAY", "MO", "FR"))


_WEB = re.compile(r"web\.(?:run|search)|to=web|<tool_call>\s*web|\bweb\s*\(|search\(|search_query", re.I)

CASES = [
    Case("gpt5_reminder", "ChatGPT 5.md", "Remind me every weekday at 9am to submit my timesheet.",
         {"vevent": has("BEGIN:VEVENT"), "rrule_weekdays": gpt5_rrule}, "description"),
    Case("gpt5_bio", "ChatGPT 5.md", "Please remember that I prefer metric units and my name is Priya.",
         {"uses_bio_tool": lambda r: re.search(r"\bbio\b", r, re.I) is not None, "stores_name": has("priya")},
         "when to use the bio tool"),
    Case("gpt5_web", "ChatGPT 5.md", "Who won the most recent Formula 1 Grand Prix?",
         {"uses_web_tool": lambda r: _WEB.search(r) is not None}, "web"),
    Case("v0_react", "V0.md", "Create a simple login form component with an email and a password field.",
         {"react_block": has('type="react"'), "shadcn_import": has("@/components/ui")}, "structure"),
    Case("v0_markdown", "V0.md", "Write me a short README for a to-do app.",
         {"markdown_block": has('type="markdown"')}, "markdown"),
    Case("v0_quiz", "V0.md", "Quiz me on React hooks.",
         {"quiz_component": has("<Quiz")}, "quiz"),
    Case("v0_node", "V0.md", "Run some Node.js code that prints the first 5 Fibonacci numbers.",
         {"nodejs_block": has('type="nodejs"')}, "nodejs_executable"),
]
ARMS = ["orig1", "ctrl_deleted", "A", "B2"]


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def run(cases: list[Case], client: LLMClient) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.jsonl"
    done = {(r["case"], r["arm"], r["rep"]) for r in _load(log) if not r.get("error")}
    red = PromptReducer()

    def go(case: Case, arm: str, rep: int, prompt: str, text: str, info: dict) -> dict | None:
        if (case.id, arm, rep) in done:
            return next(r for r in _load(log) if (r["case"], r["arm"], r["rep"]) == (case.id, arm, rep) and not r.get("error"))
        row = {"case": case.id, "arm": arm, "rep": rep, "chars": len(prompt), "info": info,
               "saved_pct": round(100 * (1 - len(prompt) / len(text)), 1)}
        try:
            reply = client.chat("agent", [{"role": "system", "content": prompt}, {"role": "user", "content": case.task}],
                                models=(client.config.agent.models[0],), attempts=4)
            row.update(model=reply.model, secs=reply.secs, response=reply.text[:1500],
                       passed={n: bool(f(reply.text)) for n, f in case.checks.items()})
        except DailyLimit:
            raise
        except LLMError as e:
            row["error"] = str(e)[:150]
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" not in row:
            done.add((case.id, arm, rep))
        print(f"  {case.id:14} {arm:13} rep{rep} {row['chars']:6} chars saved {row['saved_pct']:5.1f}%  "
              f"{sum(row.get('passed', {}).values())}/{len(case.checks)} {row.get('error', '')}", flush=True)
        return row if "error" not in row else None

    for case in cases:
        text = (PROMPTS / case.file).read_text(encoding="utf-8")
        a = red.reduce(text, task=case.task)
        drop, b2info = select_b2(text, case.task, client) if (case.id, "B2", 1) not in done else ([], {})
        b2 = red.reduce(text, task=case.task, drop_modules=drop)
        b2info["dropped"] = [m["name"] for m in b2["metadata"]["modules"] if not m["kept"]]
        b2info["restored_by_guards"] = [f"{m['name']} ({m['reason']})" for m in b2["metadata"]["modules"]
                                        if m["id"] in drop and m["kept"] and "selector" not in m["reason"]]
        prompts = {"orig1": (text, {}), "ctrl_deleted": (delete_module(text, case.key_module), {}),
                   "A": (a["context"], {"dropped": [m["name"] for m in a["metadata"]["modules"] if not m["kept"]]}),
                   "B2": (b2["context"], b2info)}
        print(f"\n>>> {case.id}", flush=True)
        (OUT / case.id).mkdir(exist_ok=True)
        got = {}
        for arm in ARMS:
            prompt, info = prompts[arm]
            (OUT / case.id / f"{arm}.txt").write_text(prompt, encoding="utf-8")
            got[arm] = go(case, arm, 1, prompt, text, info)
        o1 = got.get("orig1")
        if o1 and all(o1["passed"].values()):
            for arm in ("A", "B2"):
                r = got.get(arm)
                if r and r["passed"] != o1["passed"]:
                    go(case, arm, 2, prompts[arm][0], text, prompts[arm][1])
                    go(case, "orig1", 2, text, text, {})


def report() -> None:
    rows = [r for r in _load(OUT / "results.jsonl") if not r.get("error")]
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["case"], {}).setdefault(r["arm"], {})[r["rep"]] = r
    allc = {c.id: c for c in CASES}
    print("\n" + "=" * 100)
    tally = {"A": [], "B2": []}
    for cid, arms in by.items():
        o1 = arms.get("orig1", {}).get(1)
        ctrl = arms.get("ctrl_deleted", {}).get(1)
        if not o1:
            continue
        orig_ok = all(o1["passed"].values())
        ctrl_fails = bool(ctrl) and not all(ctrl["passed"].values())
        valid = orig_ok and ctrl_fails
        why = "" if valid else ("original fails its own checks" if not orig_ok else "control did not fail (checks can't see the section)")
        print(f"\n{cid} [{'VALID' if valid else 'EXCLUDED: ' + why}]  {allc[cid].task[:70]}")
        for arm in ARMS:
            for rep, r in sorted(arms.get(arm, {}).items()):
                same = r["passed"] == o1["passed"]
                print(f"  {arm + ('#' + str(rep) if rep > 1 else ''):13} saved {r['saved_pct']:5.1f}%  {r['passed']}  "
                      f"{'' if arm == 'orig1' else ('same as original' if same else 'DIFFERS')}")
        for arm in ("A", "B2"):
            if valid and arm in arms:
                r1 = arms[arm][1]
                held = all(r["passed"] == o1["passed"] for r in arms[arm].values())
                tally[arm].append((cid, r1["saved_pct"], held))
        if valid and "B2" in arms:
            i = arms["B2"][1]["info"]
            print(f"      B2 dropped: {i.get('dropped')}")
            if i.get("restored_by_guards"):
                print(f"      guards restored: {i['restored_by_guards'][:4]}")
    print("\n" + "=" * 100 + "\nSUMMARY (valid cases only)")
    for arm, t in tally.items():
        if t:
            held = sum(h for _, _, h in t)
            print(f"  {arm:3} held {held}/{len(t)}   saved: " + ", ".join(f"{c}={s:.0f}%{'' if h else '(BROKE)'}" for c, s, h in t)
                  + f"   mean {sum(s for _, s, _ in t) / len(t):.0f}%")


def main(argv: list[str]) -> None:
    client = LLMClient()
    try:
        run([c for c in CASES if not argv or c.id in argv], client)
    except (BudgetExceeded, DailyLimit) as e:
        print("STOPPED:", e, "-- rerun the same command to resume.")
    report()
    print(f"\nLLM calls this run: {client.calls}")


if __name__ == "__main__":
    main(sys.argv[1:])
