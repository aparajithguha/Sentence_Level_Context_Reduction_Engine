"""
prompt_structural_check.py
==========================
No-LLM check of the redundancy-only prompt reducer (``PromptReducer().reduce(text)``, v1) on a corpus of
real system prompts: does it break anything, does it keep every instruction, and how much does it remove?

Per prompt it checks that the reduced text is never bigger, keeps code fences balanced, does not unbalance
tag pairs, adds no ``\\uXXXX`` escapes, and keeps every rule line (a line with must / never / always /
do not / ...) word for word. Missing rule lines are traced to the block that held them and the reason
that block was dropped.

The corpus is third-party text and is not in the repository. Expected folders (see README):
  System-Prompts-main/*.md
  prompt_datasets/piebald_claude_code/system-prompts/*.md
  prompt_datasets/hf_system_prompt_library/system-prompts/json/*.json

Run: python -m tests.prompt_structural_check
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics
import time
from pathlib import Path

from scre.prompt_reducer import PromptReducer

ROOT = Path(__file__).resolve().parent.parent
RULE_RE = re.compile(r"\b(must|never|always|do not|don't|should not|shouldn't|forbidden|prohibited|required)\b", re.I)
_TAG = re.compile(r"<(/?)([A-Za-z_][\w\-]*)(?:\s[^<>]*)?>")


def load_corpus() -> list[dict]:
    items = []
    for p in sorted(glob.glob(str(ROOT / "System-Prompts-main/*.md"))):
        if os.path.basename(p) != "README.md":
            items.append({"source": "full agent prompts", "text": open(p, encoding="utf-8", errors="replace").read()})
    for p in sorted(glob.glob(str(ROOT / "prompt_datasets/piebald_claude_code/system-prompts/*.md"))):
        t = re.sub(r"\A\s*<!--.*?-->\s*", "", open(p, encoding="utf-8", errors="replace").read(), count=1, flags=re.S)
        items.append({"source": "Claude Code fragments", "text": t})
    for p in sorted(glob.glob(str(ROOT / "prompt_datasets/hf_system_prompt_library/system-prompts/json/*.json"))):
        try:
            t = (json.load(open(p, encoding="utf-8")).get("systemprompt") or "")
        except (ValueError, OSError):
            continue
        if t.strip():
            items.append({"source": "community prompt library", "text": t})
    return items


def _norm(s: str) -> str:
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def rule_lines(text: str) -> list[str]:
    out, fence = [], False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        elif not fence and 15 <= len(s) <= 400 and RULE_RE.search(s):
            out.append(s)
    return out


def _fence_lines(t: str) -> int:
    return sum(1 for l in t.split("\n") if l.lstrip().startswith(("```", "~~~")))


def _tag_imbalance(t: str) -> int:
    opens, closes = collections.Counter(), collections.Counter()
    for m in _TAG.finditer(t):
        (closes if m.group(1) else opens)[m.group(2).lower()] += 1
    return sum(abs(opens[k] - closes[k]) for k in set(opens) | set(closes))


def bucket(n: int) -> str:
    return "under 2K chars" if n < 2000 else ("2K-8K chars" if n < 8000 else "8K chars and over")


def main() -> None:
    corpus = load_corpus()
    if not corpus:
        raise SystemExit("no prompts found: see the folder list in this file's docstring")
    red = PromptReducer()
    fails = collections.Counter()
    by_bucket: dict[str, list] = collections.defaultdict(list)
    by_source = collections.Counter()
    rules_total = rules_kept = 0
    missing_reason = collections.Counter()
    t0 = time.time()
    for it in corpus:
        text = it["text"]
        r = red.reduce(text)
        out = r["context"]
        by_source[it["source"]] += 1
        if len(out) > len(text):
            fails["output bigger than input"] += 1
        if _fence_lines(text) % 2 == 0 and _fence_lines(out) % 2 != 0:
            fails["unbalanced code fence"] += 1
        if _tag_imbalance(out) > _tag_imbalance(text):
            fails["tag pairs unbalanced"] += 1
        if len(re.findall(r"\\u[0-9a-fA-F]{4}", out)) > len(re.findall(r"\\u[0-9a-fA-F]{4}", text)):
            fails["new \\u escapes"] += 1
        nout = _norm(out)
        dropped = [(_norm(b["text"]), b["reason"]) for b in r["blocks"] if not b["kept"]]
        for line in rule_lines(text):
            rules_total += 1
            key = _norm(line)[:40]
            if key in nout:
                rules_kept += 1
            else:
                why = next((reason.split(" of block")[0] for t, reason in dropped if key in t), "other")
                missing_reason[why] += 1
        by_bucket[bucket(len(text))].append((len(text), len(text) - len(out), r["metadata"]["kept_blocks"] < r["metadata"]["blocks"]))
    print(f"{len(corpus)} prompts in {time.time() - t0:.1f}s | sources: {dict(by_source)}")
    print(f"integrity failures: {dict(fails) or 'none'}")
    print(f"rule lines kept word for word: {rules_kept}/{rules_total}; the {rules_total - rules_kept} missing were in: {dict(missing_reason)}")
    print(f"{'size bucket':20}{'prompts':>8}{'text changed':>13}{'blocks removed':>15}{'median saved':>14}{'mean saved':>12}{'saved (by chars)':>18}")
    groups = [(b, by_bucket[b]) for b in ("under 2K chars", "2K-8K chars", "8K chars and over")]
    groups.append(("all", [x for v in by_bucket.values() for x in v]))
    for name, rows in groups:
        ratios = [s / n for n, s, _ in rows]
        print(f"{name:20}{len(rows):>8}{sum(1 for _, s, _ in rows if s > 0):>13}{sum(1 for *_, d in rows if d):>15}"
              f"{100 * statistics.median(ratios):>13.1f}%{100 * statistics.mean(ratios):>11.1f}%"
              f"{100 * sum(s for _, s, _ in rows) / sum(n for n, _, _ in rows):>17.1f}%")


if __name__ == "__main__":
    main()
