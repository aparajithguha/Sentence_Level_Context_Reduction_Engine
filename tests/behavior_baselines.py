"""
behavior_baselines.py
=====================
Baselines at the SAME size as B2 (with guards) on the 8 valid cases, to see whether the size cut alone,
or a standard selection method, explains B2's result. Every arm is cut to B2's character count.

  truncation   keep the start of the prompt, cut at a line boundary
  random       drop random whole sections; no guards, no LLM
  bm25         rank sections by BM25 against the task, drop the lowest-ranked; no guards, no LLM
  vector       same with sentence-embedding cosine similarity (all-mpnet-base-v2); no guards
  selective    Selective Context (Li et al. 2023): GPT-2 self-information per sentence, drop the most
               predictable sentences first; task-agnostic, ignores structure, as published

random / bm25 / vector choose among the same sections (non-global, >=400 chars) and keep everything else, so they
inherit our parser and the global tier -- a generous setup that isolates the selection signal. Selective Context
gets no structure at all. A case's original passes all its valid checks (that is what made it valid), so "held" =
every valid check passes. An arm that fails is re-run once to separate drift from noise.

Run: python -m tests.behavior_baselines [arm ...] [case_id ...]      e.g.  bm25 vector
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # models are used from the local cache only

from scre.blocks import parse_blocks, rebuild
from scre.config import PromptConfig
from scre.prompt_modules import find_modules
from scre.roles import classify
from tests.behavior_pilot import PROMPTS, Case
from tests.behavior_round3 import CASES as ROUND123, VALID as VALID3
from tests.behavior_round4 import CASES as ROUND4
from tests.llm_config import BudgetExceeded, DailyLimit, LLMClient, LLMError

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "prompt_datasets" / "results"
OUT = RES / "behavior_baselines"

CASES: dict[str, Case] = {c.id: c for c in list(ROUND123.values()) + ROUND4}
VALID = dict(VALID3, gpt5_bio=["uses_bio_tool", "stores_name"], gpt5_web=["uses_web_tool"],
             v0_markdown=["markdown_block"], v0_quiz=["quiz_component"])
B2_FILES = {
    "codex_hello": RES / "behavior_round3/codex_hello_B2.txt", "gemini_newapp": RES / "behavior_round3/gemini_newapp_B2.txt",
    "gpt41_image": RES / "behavior_round3/gpt41_image_B2.txt", "gpt5_web": RES / "behavior_round4/gpt5_web/B2.txt",
    "v0_markdown": RES / "behavior_round4/v0_markdown/B2.txt", "v0_quiz": RES / "behavior_round4/v0_quiz/B2.txt",
}
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
GPT2 = "openai-community/gpt2"


def b2_prompt(cid: str) -> str:
    """The final B2 prompt (with all guards) for a case."""
    if cid in ("gpt5_bio", "gpt41_chart"):                   # rebuilt with the fixpoint guards
        return json.load(open(RES / "guard_fixpoint_prompts.json"))[cid]
    return B2_FILES[cid].read_text(encoding="utf-8")


# ----------------------------------------------------------------- arms: fn(text, target_chars, task) -> prompt
def truncate(text: str, target: int, task: str = "") -> str:
    cut = text[:target]
    nl = cut.rfind("\n")
    return (cut[: nl + 1] if nl > 0 else cut).rstrip()


def _pool(text: str):
    blocks = parse_blocks(text)
    classify(blocks)
    cfg = PromptConfig()
    return blocks, [m for m in find_modules(blocks, cfg) if m.cls != "global" and m.chars >= cfg.module_min_chars]


def _drop_in_order(blocks, order, target: int, fill: bool = False) -> str:
    """Drop sections in ``order`` until the prompt fits ``target``. With ``fill``, then put back dropped sections
    (best-ranked first) while the prompt still fits, so whole-section drops do not undershoot the size by much."""
    keep = {b.id for b in blocks}
    out = rebuild(blocks, keep)
    dropped = []
    for m in order:
        if len(out) <= target:
            break
        keep -= set(m.block_ids)
        dropped.append(m)
        out = rebuild(blocks, keep)
    if fill:
        for m in reversed(dropped):
            trial = keep | set(m.block_ids)
            if len(rebuild(blocks, trial)) <= target:
                keep = trial
        out = rebuild(blocks, keep)
    return out.rstrip()


def random_sections(text: str, target: int, task: str = "", seed: int = 7) -> str:
    blocks, pool = _pool(text)
    random.Random(seed).shuffle(pool)
    return _drop_in_order(blocks, pool, target)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25_sections(text: str, target: int, task: str) -> str:
    from rank_bm25 import BM25Okapi
    blocks, pool = _pool(text)
    scores = BM25Okapi([_words(m.body) for m in pool]).get_scores(_words(task))
    order = [m for _, _, m in sorted(zip(scores, [-m.chars for m in pool], pool), key=lambda t: (t[0], t[1]))]  # lowest first
    return _drop_in_order(blocks, order, target, fill=True)


_embedder = None


def vector_sections(text: str, target: int, task: str) -> str:
    global _embedder
    from sentence_transformers import SentenceTransformer, util
    _embedder = _embedder or SentenceTransformer(EMBED_MODEL)
    blocks, pool = _pool(text)
    q = _embedder.encode(task, convert_to_tensor=True)
    d = _embedder.encode([m.body[:2000] for m in pool], convert_to_tensor=True)
    scores = util.cos_sim(q, d)[0].tolist()
    order = [m for _, _, m in sorted(zip(scores, [-m.chars for m in pool], pool), key=lambda t: (t[0], t[1]))]
    return _drop_in_order(blocks, order, target, fill=True)


_gpt2 = None


def _token_self_info(text: str) -> tuple[list[tuple[int, int]], list[float]]:
    """-log p(token | previous tokens) with GPT-2, in non-overlapping 1000-token windows."""
    global _gpt2
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if _gpt2 is None:
        _gpt2 = (AutoTokenizer.from_pretrained(GPT2), AutoModelForCausalLM.from_pretrained(GPT2).eval())
    tok, model = _gpt2
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    info: list[float] = []
    for s in range(0, len(ids), 1000):
        chunk = ids[s:s + 1000]
        inp = torch.tensor([[tok.bos_token_id] + chunk])
        with torch.no_grad():
            logp = torch.log_softmax(model(inp).logits[0][:-1].float(), dim=-1)
        info += (-logp[torch.arange(len(chunk)), torch.tensor(chunk)]).tolist()
    return offs, info


def _sentence_units(text: str) -> list[tuple[int, int]]:
    """Character spans that tile the text: each line, split further after sentence-ending punctuation."""
    spans = []
    for line in re.finditer(r"[^\n]*\n|[^\n]+$", text):
        s, e = line.start(), line.end()
        cut = s
        for m in re.finditer(r"(?<=[.!?])[ \t]+", line.group(0)):
            spans.append((cut, s + m.end()))
            cut = s + m.end()
        if cut < e:
            spans.append((cut, e))
    return spans or [(0, len(text))]


def selective_context(text: str, target: int, task: str = "") -> str:
    offs, info = _token_self_info(text)
    units = _sentence_units(text)
    score = [0.0] * len(units)
    ui = 0
    for (a, _), i in zip(offs, info):                       # each token counts toward the unit holding its start
        while ui < len(units) - 1 and a >= units[ui][1]:
            ui += 1
        score[ui] += i
    keep = set(range(len(units)))
    size = len(text)
    for u in sorted(range(len(units)), key=lambda k: score[k]):   # least informative (most predictable) first
        if size <= target:
            break
        keep.discard(u)
        size -= units[u][1] - units[u][0]
    return "".join(text[units[k][0]:units[k][1]] for k in sorted(keep)).rstrip()


ARMS = {"truncation": truncate, "random": random_sections, "bm25": bm25_sections,
        "vector": vector_sections, "selective": selective_context}


# ----------------------------------------------------------------- run
def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def run(client: LLMClient, arms: list[str], only: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.jsonl"
    done = {(r["case"], r["arm"], r["rep"]) for r in _load(log) if not r.get("error")}

    def go(case: Case, arm: str, rep: int, prompt: str, text: str) -> dict | None:
        if (case.id, arm, rep) in done:
            return next(r for r in _load(log) if (r["case"], r["arm"], r["rep"]) == (case.id, arm, rep) and not r.get("error"))
        row = {"case": case.id, "arm": arm, "rep": rep, "chars": len(prompt), "saved_pct": round(100 * (1 - len(prompt) / len(text)), 1)}
        try:
            reply = client.chat("agent", [{"role": "system", "content": prompt}, {"role": "user", "content": case.task}],
                                models=(client.config.agent.models[0],), attempts=4)
            row.update(model=reply.model, response=reply.text[:600],
                       passed={n: bool(f(reply.text)) for n, f in case.checks.items() if n in VALID[case.id]})
        except DailyLimit:
            raise
        except LLMError as e:
            row["error"] = str(e)[:150]
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" not in row:
            done.add((case.id, arm, rep))
        print(f"  {case.id:14} {arm:11} rep{rep} {row['chars']:6} chars saved {row['saved_pct']:5.1f}%  "
              f"{sum(row.get('passed', {}).values())}/{len(VALID[case.id])} {row.get('error', '')}", flush=True)
        return row if "error" not in row else None

    for cid in VALID:
        if only and cid not in only:
            continue
        case = CASES[cid]
        text = (PROMPTS / case.file).read_text(encoding="utf-8")
        target = len(b2_prompt(cid))
        print(f"\n>>> {cid}  (B2 size {target} chars, {100 * (1 - target / len(text)):.1f}% saved)", flush=True)
        for arm in arms:
            prompt = ARMS[arm](text, target, case.task)
            (OUT / f"{cid}_{arm}.txt").write_text(prompt, encoding="utf-8")
            r = go(case, arm, 1, prompt, text)
            if r and not all(r["passed"].values()):
                go(case, arm, 2, prompt, text)


def report() -> None:
    rows = [r for r in _load(OUT / "results.jsonl") if not r.get("error")]
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["case"], {}).setdefault(r["arm"], {})[r["rep"]] = r
    arms = [a for a in ARMS if any(a in v for v in by.values())]
    print("\n" + "=" * (18 + 22 * len(arms)))
    print(f"{'case':14}{'B2':>6}   " + "".join(f"{a:<22}" for a in arms))
    tally: dict[str, list] = {a: [] for a in arms}
    for cid in VALID:
        if cid not in by:
            continue
        b2 = 100 * (1 - len(b2_prompt(cid)) / len((PROMPTS / CASES[cid].file).read_text(encoding="utf-8")))
        cells = []
        for arm in arms:
            runs = by[cid].get(arm, {})
            if not runs:
                cells.append("-")
                continue
            held = all(all(r["passed"].values()) for r in runs.values())
            tally[arm].append((runs[1]["saved_pct"], held))
            n_ok = sum(all(r["passed"].values()) for r in runs.values())
            cells.append(f"{runs[1]['saved_pct']:3.0f}% {'HELD' if held else 'BROKE'}" + ("" if len(runs) == 1 else f" ({n_ok}/{len(runs)})"))
        print(f"{cid:14}{b2:5.0f}%   " + "".join(f"{c:<22}" for c in cells))
    print()
    for arm, t in tally.items():
        if t:
            print(f"  {arm:11} held {sum(h for _, h in t)}/{len(t)}   mean saved {sum(s for s, _ in t) / len(t):.0f}%")
    print("  B2+guards   held 8/8 (rounds 3-4 and the guard rerun)   mean saved 56%")


def main(argv: list[str]) -> None:
    arms = [a for a in argv if a in ARMS] or list(ARMS)
    client = LLMClient()
    try:
        run(client, arms, [a for a in argv if a not in ARMS])
    except (BudgetExceeded, DailyLimit) as e:
        print("STOPPED:", e, "-- rerun the same command to resume.")
    report()
    print(f"\nLLM calls this run: {client.calls}")


if __name__ == "__main__":
    main(sys.argv[1:])
