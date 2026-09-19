"""
behavior_round2.py
==================
Round 2: only the approaches that held in Round 1, on new prompts.

  orig1          original prompt
  ctrl_deleted   original minus the section holding the key rule -> must fail a check, or the case is invalid
  A              deterministic task matcher (no LLM)
  C              LLM-tightened wording, each rewrite verified by code (one LLM call per prompt, reused)
  AC             A's kept sections with C's verified rewrites applied

Whenever an approach's checks differ from the original's, it and the original are re-run once,
to tell real drift from run-to-run noise. Resumable: finished (case, arm, rep) rows are skipped.

Run: python -m tests.behavior_round2 [case_id ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scre.prompt_reducer import PromptReducer
from tests.behavior_pilot import CASES as ROUND1_CASES, PROMPTS, Case, apply_rewrites, delete_module, tighten_rewrites
from tests.llm_config import BudgetExceeded, DailyLimit, LLMClient, LLMError

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "prompt_datasets" / "results" / "behavior_round2"


# ----------------------------------------------------------------- checks (written before any run)
def gem_git_gather(r: str) -> bool:
    l = r.lower()
    return "git status" in l and "git diff" in l and re.search(r"git log\s+-n\s*3", l) is not None


def gem_bootstrap(r: str) -> bool:
    return "bootstrap" in r.lower()


def gem_material(r: str) -> bool:
    return "material" in r.lower()


def gpt_image_tool(r: str) -> bool:
    l = r.lower()
    return "image_gen" in l or "text2im" in l


def gpt_no_question(r: str) -> bool:
    return "?" not in r


def gpt_mpl_no_seaborn(r: str) -> bool:
    l = r.lower()
    return "matplotlib" in l and "seaborn" not in l


_COLOR = re.compile(r"\b(?:color|colors|facecolor|edgecolor|cmap)\s*=|plt\.style|style\.use|,\s*['\"][bgrcmykw][-:.o]+['\"]")


def gpt_no_colors(r: str) -> bool:
    return "plot" in r.lower() and _COLOR.search(r) is None


CASES = [
    Case("gemini_commit", "Gemini CLI.md", "Commit my staged changes to the repo.",
         {"git_status_diff_log3": gem_git_gather}, "git repository"),
    Case("gemini_newapp", "Gemini CLI.md",
         "Build me a small web app that shows a to-do list. Don't ask me questions: choose the tech stack yourself "
         "and just tell me the plan.",
         {"bootstrap": gem_bootstrap, "material_design": gem_material}, "new applications"),
    Case("gpt41_image", "ChatGPT 4.1.md", "Draw me a picture of a red bicycle leaning on a brick wall.",
         {"uses_image_gen": gpt_image_tool, "no_clarifying_question": gpt_no_question}, "imagegen"),
    Case("gpt41_chart", "ChatGPT 4.1.md",
         "Monthly sales: Jan 10, Feb 14, Mar 9, Apr 20. Make me a line chart of it.",
         {"matplotlib_no_seaborn": gpt_mpl_no_seaborn, "no_colors_set": gpt_no_colors}, "python"),
]
ARMS = ["orig1", "ctrl_deleted", "A", "C", "AC"]


# ----------------------------------------------------------------- run
def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def _rewrites(text: str, file: str, client: LLMClient) -> tuple[dict[int, str], dict]:
    cache = OUT / f"rewrites_{Path(file).stem.replace(' ', '_')}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return {int(k): v for k, v in d["rewrites"].items()}, d["info"]
    rewrites, info = tighten_rewrites(text, client)
    cache.write_text(json.dumps({"rewrites": rewrites, "info": info}))
    return rewrites, info


def run(cases: list[Case], arms_by_case: dict[str, list[str]], client: LLMClient) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.jsonl"
    done = {(r["case"], r["arm"], r["rep"]) for r in _load(log) if not r.get("error")}
    red = PromptReducer()

    def go(case: Case, arm: str, rep: int, prompt: str, text: str) -> None:
        if (case.id, arm, rep) in done:
            return
        row = {"case": case.id, "arm": arm, "rep": rep, "chars": len(prompt), "saved_pct": round(100 * (1 - len(prompt) / len(text)), 1)}
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
        done.add((case.id, arm, rep)) if "error" not in row else None
        print(f"  {case.id:14} {arm:13} rep{rep} {row['chars']:6} chars saved {row['saved_pct']:5.1f}%  "
              f"{sum(row.get('passed', {}).values())}/{len(case.checks)} {row.get('error', '')}", flush=True)

    for case in cases:
        text = (PROMPTS / case.file).read_text(encoding="utf-8")
        wanted = arms_by_case.get(case.id, ARMS)
        need_c = any(a in wanted for a in ("C", "AC"))
        rewrites, info = _rewrites(text, case.file, client) if need_c else ({}, {})
        a = red.reduce(text, task=case.task)
        prompts = {"orig1": text, "ctrl_deleted": delete_module(text, case.key_module), "A": a["context"],
                   "C": apply_rewrites(red.reduce(text)["blocks"], rewrites), "AC": apply_rewrites(a["blocks"], rewrites)}
        print(f"\n>>> {case.id}  (rewrites accepted {info.get('rewritten_ok', '-')}/{info.get('blocks_sent', '-')}; "
              f"A dropped: {[m['name'] for m in a['metadata']['modules'] if not m['kept']]})", flush=True)
        (OUT / case.id).mkdir(exist_ok=True)
        for arm in wanted:
            (OUT / case.id / f"{arm}.txt").write_text(prompts[arm], encoding="utf-8")
            go(case, arm, 1, prompts[arm], text)
        # confirmation reruns: an approach that differs from the original is re-run with the original
        rows = [r for r in _load(log) if r["case"] == case.id and not r.get("error")]
        o1 = next((r for r in rows if r["arm"] == "orig1" and r["rep"] == 1), None)
        for r in rows:
            if o1 and r["arm"] in ("A", "C", "AC") and r["rep"] == 1 and r["passed"] != o1["passed"]:
                go(case, "orig1", 2, text, text)
                go(case, r["arm"], 2, prompts[r["arm"]], text)


def report() -> None:
    rows = [r for r in _load(OUT / "results.jsonl") if not r.get("error")]
    allcases = {c.id: c for c in CASES + ROUND1_CASES}
    print("\n" + "=" * 100)
    for cid in dict.fromkeys(r["case"] for r in rows):
        case = allcases[cid]
        by = {}
        for r in rows:
            if r["case"] == cid:
                by.setdefault(r["arm"], {})[r["rep"]] = r
        o1 = by.get("orig1", {}).get(1)
        print(f"\n{cid}: {case.task[:80]}")
        print(f"  {'arm':13}{'chars':>7}{'saved':>7}  " + "  ".join(f"{n[:16]:16}" for n in case.checks) + "  verdict")
        for arm in ARMS:
            for rep, r in sorted(by.get(arm, {}).items()):
                cells = "  ".join(f"{'PASS' if r['passed'][n] else 'FAIL':16}" for n in case.checks)
                if arm == "orig1":
                    verdict = "" if rep == 1 else ("same as orig run 1" if r["passed"] == o1["passed"] else "orig differs from itself: NOISE")
                elif arm == "ctrl_deleted":
                    verdict = "control fails (case valid)" if o1 and r["passed"] != o1["passed"] and not all(r["passed"].values()) else "CONTROL DID NOT FAIL: case insensitive"
                else:
                    verdict = "same as original" if o1 and r["passed"] == o1["passed"] else "DIFFERS from original"
                print(f"  {arm + ('#' + str(rep) if rep > 1 else ''):13}{r['chars']:>7}{r['saved_pct']:>6.1f}%  {cells}  {verdict}")


def main(argv: list[str]) -> None:
    client = LLMClient()
    cases = [c for c in CASES if not argv or c.id in argv]
    arms_by_case = {}
    if not argv or "codex_hello" in argv:                    # Round 1 already covered everything but A+C
        codex = next(c for c in ROUND1_CASES if c.id == "codex_hello")
        cases.append(codex)
        arms_by_case["codex_hello"] = ["AC"]
    try:
        run(cases, arms_by_case, client)
    except (BudgetExceeded, DailyLimit) as e:
        print("STOPPED:", e, "-- finished rows are kept; rerun the same command to resume.")
    report()
    print(f"\nLLM calls this run: {client.calls}")


if __name__ == "__main__":
    main(sys.argv[1:])
