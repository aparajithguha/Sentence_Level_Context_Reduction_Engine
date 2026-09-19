"""
behavior_round3.py
==================
Round 3: approach B fixed ("B2"), tested on the cases that were valid in rounds 1-2.

B2 = an LLM picks sections to drop, but
  * it sees each section's full text (Round 1 showed it only a 200-character preview);
  * it must first list the steps the agent will take and which sections each step needs;
    a section used by any step is never dropped;
  * code then applies two cross-reference guards from the prompt's own text
    (see scre.prompt_modules): keep a section whose identifier the kept text uses, and keep a
    section that defines vocabulary the prompt's own examples use.

Arms:
  B_guarded   Round 1's B picks for Codex, replayed through the guards (no new LLM reducer call)
  B2          fresh selection as above

Reference results (original prompt) come from rounds 1-2. Only checks that were valid there count.
An arm that differs from the original is re-run once, with the original, to separate drift from noise.

Run: python -m tests.behavior_round3 [case_id ...]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from scre.prompt_reducer import PromptReducer
from scre.prompt_select import llm_selector
from tests.behavior_pilot import CASES as ROUND1_CASES, PROMPTS, Case
from tests.behavior_round2 import CASES as ROUND2_CASES
from tests.llm_config import BudgetExceeded, DailyLimit, LLMClient, LLMError

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "prompt_datasets" / "results"
OUT = RESULTS / "behavior_round3"

CASES = {c.id: c for c in ROUND1_CASES + ROUND2_CASES}
VALID = {  # case -> checks that were sensitive in rounds 1-2 (original passes, control fails)
    "codex_hello": ["begin_patch", "add_file_hello"],
    "gemini_newapp": ["bootstrap", "material_design"],
    "gpt41_image": ["uses_image_gen", "no_clarifying_question"],
    "gpt41_chart": ["matplotlib_no_seaborn"],
}


def select_b2(text: str, task: str, client: LLMClient) -> tuple[list[int], dict]:
    """B2 through the library's own selector (``scre.prompt_select``); the LLM call is the harness's."""
    meta: dict = {}

    def ask(messages: list[dict]) -> str:
        reply = client.chat("reducer", messages, max_tokens=4000)
        meta.update(reducer_model=reply.model, reducer_secs=reply.secs)
        return reply.text

    selector = llm_selector(ask)
    outline = PromptReducer().outline(text, max_text_chars=selector.max_text_chars)
    drop = selector(outline, task)
    return list(drop), {"steps": selector.last["steps"], "llm_drop": selector.last["proposed_drop"],
                        "kept_because_a_step_needs_it": selector.last["kept_because_a_step_needs_it"], **meta}


def _reference(case_id: str) -> dict[str, bool] | None:
    """The original prompt's check results, restricted to the valid checks."""
    rows = []
    p2 = RESULTS / "behavior_round2" / "results.jsonl"
    if p2.exists():
        rows += [json.loads(l) for l in p2.read_text().splitlines()]
    for r in rows:
        if r["case"] == case_id and r["arm"] == "orig1" and r["rep"] == 1 and not r.get("error"):
            return {k: v for k, v in r["passed"].items() if k in VALID[case_id]}
    if case_id == "codex_hello":
        r1 = json.load(open(sorted(glob.glob(str(RESULTS / "behavior_round1" / "results_*.json")))[-1]))
        o = next(r for r in r1[case_id] if r["arm"] == "orig1")
        return {k: v for k, v in o["passed"].items() if k in VALID[case_id]}
    return None


def run(case_ids: list[str], client: LLMClient) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    done = {(r["case"], r["arm"], r["rep"]) for r in rows if not r.get("error")}
    red = PromptReducer()

    def go(case: Case, arm: str, rep: int, prompt: str, text: str, info: dict) -> dict | None:
        if (case.id, arm, rep) in done:
            return next(r for r in rows if (r["case"], r["arm"], r["rep"]) == (case.id, arm, rep) and not r.get("error"))
        row = {"case": case.id, "arm": arm, "rep": rep, "chars": len(prompt), "info": info,
               "saved_pct": round(100 * (1 - len(prompt) / len(text)), 1)}
        try:
            reply = client.chat("agent", [{"role": "system", "content": prompt}, {"role": "user", "content": case.task}],
                                models=(client.config.agent.models[0],), attempts=4)
            row.update(model=reply.model, secs=reply.secs, response=reply.text[:1500],
                       passed={n: bool(f(reply.text)) for n, f in case.checks.items() if n in VALID[case.id]})
        except DailyLimit:
            raise
        except LLMError as e:
            row["error"] = str(e)[:150]
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        if "error" not in row:
            done.add((case.id, arm, rep))
        ok = sum(row.get("passed", {}).values())
        print(f"  {case.id:14} {arm:11} rep{rep} {row['chars']:6} chars saved {row['saved_pct']:5.1f}%  {ok}/{len(VALID[case.id])} {row.get('error', '')}", flush=True)
        return row if "error" not in row else None

    for cid in case_ids:
        case = CASES[cid]
        text = (PROMPTS / case.file).read_text(encoding="utf-8")
        ref = _reference(cid)
        print(f"\n>>> {cid}  (original passes: {ref})", flush=True)
        arms: list[tuple[str, str, dict]] = []
        if cid == "codex_hello":
            r1 = json.load(open(sorted(glob.glob(str(RESULTS / "behavior_round1" / "results_*.json")))[-1]))
            b = next(r for r in r1[cid] if r["arm"] == "B")
            out = red.outline(text)
            ids: list[int] = []
            for n in b["info"]["asked_to_drop"]:
                ids.append(next(m["id"] for m in out if m["name"] == n and m["id"] not in ids))
            res = red.reduce(text, task=case.task, drop_modules=ids)
            arms.append(("B_guarded", res["context"], {"restored_by_guards": [m["name"] for m in res["metadata"]["modules"]
                                                                            if m["id"] in ids and m["kept"] and "selector" not in m["reason"]]}))
        drop, info = select_b2(text, case.task, client)
        res = red.reduce(text, task=case.task, drop_modules=drop)
        info["restored_by_guards"] = [f"{m['name']} ({m['reason']})" for m in res["metadata"]["modules"]
                                      if m["id"] in drop and m["kept"] and "selector" not in m["reason"]]
        info["dropped"] = [m["name"] for m in res["metadata"]["modules"] if not m["kept"]]
        arms.append(("B2", res["context"], info))
        for arm, prompt, inf in arms:
            (OUT / f"{cid}_{arm}.txt").write_text(prompt, encoding="utf-8")
            r = go(case, arm, 1, prompt, text, inf)
            if r and ref and r["passed"] != ref:                      # differs -> confirm once
                go(case, arm, 2, prompt, text, inf)
                go(case, "orig1", 2, text, text, {})


def report() -> None:
    log = OUT / "results.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    rows = [r for r in rows if not r.get("error")]
    print("\n" + "=" * 96)
    for cid in dict.fromkeys(r["case"] for r in rows):
        ref = _reference(cid)
        print(f"\n{cid}: {CASES[cid].task[:80]}\n  original passes: {ref}")
        for r in rows:
            if r["case"] != cid or r["arm"] == "orig1":
                continue
            same = r["passed"] == ref
            print(f"  {r['arm']:10} rep{r['rep']} saved {r['saved_pct']:5.1f}%  {r['passed']}  -> {'SAME as original' if same else 'DIFFERS'}")
            i = r["info"]
            if r["rep"] == 1 and i.get("dropped") is not None:
                print(f"      dropped: {i['dropped']}")
                if i.get("restored_by_guards"):
                    print(f"      guards restored: {i['restored_by_guards']}")
                if i.get("steps"):
                    print(f"      LLM's steps: {i['steps'][:4]}")


def main(argv: list[str]) -> None:
    client = LLMClient()
    try:
        run([c for c in VALID if not argv or c in argv], client)
    except (BudgetExceeded, DailyLimit) as e:
        print("STOPPED:", e, "-- rerun the same command to resume.")
    report()
    print(f"\nLLM calls this run: {client.calls}")


if __name__ == "__main__":
    main(sys.argv[1:])
