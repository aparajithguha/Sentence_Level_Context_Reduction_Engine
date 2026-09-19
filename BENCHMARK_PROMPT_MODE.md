# Prompt-mode benchmark: does the agent behave the same on a reduced system prompt?

This report covers **prompt mode** (`reduce_prompt`, for agent system prompts). The document-mode benchmark (Kubernetes KEPs and IETF RFCs) is a different question and lives in [BENCHMARK_REPORT_sample.md](BENCHMARK_REPORT_sample.md) and the README.

Everything below was measured; nothing is estimated. Where a result is weak, small, or was tuned on the same cases, it says so. The last section lists what this benchmark cannot show.

## Summary

| Question | Answer |
|---|---|
| How much does the engine remove with **no task and no LLM**? | **2.6% of the characters** across 1,690 real prompts, **8.0%** on the 35 prompts of 8K characters or more. Nothing broke, and 1,757 of 1,765 instruction lines survived word for word (the other 8 were inside trimmed examples or a duplicate). |
| How much can it remove **with a task and an LLM choosing sections** (B2), with the agent's behavior unchanged? | **26% to 77%, mean 56%**, on 8 valid test cases. The agent's checks matched the original on **8 of 8**. |
| Does a same-size cut do the same thing? | No. At the same size as B2: truncation held on **3 of 8**, random section drop **2 of 8**, BM25 **5 of 8**, vector search **5 of 8**, Selective Context **6 of 8**. |
| Where did B2 clearly beat every baseline? | A task that needs a tool without naming it ("draw a picture of a red bicycle"). All five baselines broke it. B2 kept the image tool. |
| How sure are we? | Not very. 8 cases, one or two runs each, one agent model, and the guards were built after seeing failures on some of these same cases. B2's edge over the best baseline (6 of 8) is not statistically significant (Fisher p about 0.5). See the last section. |

## What "the agent behaves the same" means here

Each test case is a real system prompt, a task, and automatic checks on one rule the agent cannot guess (for example, that a file must be written as a patch in one exact format). We run an agent on the **original** prompt and on each **reduced** prompt and compare which checks pass.

A case only counts if it is **valid**, a rule fixed before any run:

1. the **original** prompt passes all of the case's checks, and
2. a **control**, the original with the one section that holds the key rule deleted, fails at least one check.

Rule 2 proves the checks can actually see the rule. A case whose control still passes cannot tell a good reduction from a bad one, so it is excluded.

An approach that differs from the original is re-run once to separate real drift from noise. "Held" means every valid check passed on every run.

- **Agent:** `inclusionai/ling-3.0-flash-vl:free` via OpenRouter, temperature 0.
- **LLM used by approaches B, C, D:** `nvidia/nemotron-3-ultra-550b-a55b:free`.
- The engine itself never calls an LLM. Only the test harness does (`tests/llm_config.py`).

## Corpus and the no-LLM check

1,690 real system prompts (third-party text, not in this repository; see [Reproducing](#reproducing)):

| Source | Prompts | 8K chars and over |
|---|---:|---:|
| Full agent prompts (ChatGPT, Claude Code, Codex, Gemini CLI, V0, Bolt and others) | 30 | 18 |
| Claude Code prompt fragments | 737 | 17 |
| Community prompt library | 923 | 0 |

`python -m tests.prompt_structural_check` runs the redundancy-only reducer on all of them in about 1.5 seconds:

| Size | Prompts | Text changed | Had whole blocks removed | Median saved | Mean saved | Saved (by characters) |
|---|---:|---:|---:|---:|---:|---:|
| under 2K chars | 1,339 | 746 | 24 | 0.1% | 0.2% | 0.2% |
| 2K to 8K | 316 | 195 | 35 | 0.0% | 0.6% | 0.5% |
| 8K and over | 35 | 35 | 17 | 0.1% | 4.8% | 8.0% |
| **All** | **1,690** | **976** | **76** | **0.0%** | **0.4%** | **2.6%** |

Integrity: no output was bigger than its input, no code fence or tag pair was unbalanced, and no `\uXXXX` escapes were added. Of 1,765 instruction lines (lines with must, never, always, do not and similar), **1,757 survived word for word**. The 8 that did not were inside a trimmed surplus example (7) or a duplicate whose twin remains (1).

**Why a separate mode exists.** The document-mode engine treats a prompt as a flat list of sentences and picks the most relevant ones. On these prompts it removed most of the text, but kept only about 17% to 33% of explicit rules on large prompts, flattened structure, and sometimes made the output bigger. These figures come from an earlier ad-hoc run and are not reproduced by a committed script. That is what motivated a reducer that keeps every instruction and removes only what can be shown to be redundant.

The safe, no-task reduction is small because the remaining text is mostly behavior: about 31% rules, 17% tool sections, 15% prose, 11% procedures, 8% examples and 6% code on prompts of 8K characters or more. Rules are mostly distinct, not repeated. Removing more means removing content, and which content is safe to remove depends on the task. That is what the rest of this report tests.

## The approaches

| | Approach | Uses an LLM? |
|---|---|---|
| **v1** | Redundancy only: repeated rules, surplus examples, decoration. Every other instruction stays word for word. | no |
| **A** | Drop whole sections whose words do not appear in the task (plus deterministic guards). | no |
| **B / B2** | An LLM chooses sections to drop. B saw a 200-character preview of each section. **B2** sees full section text and must first write the steps the task needs and which sections cover them; a section used by any step is never dropped. Both then pass through the guards. | yes |
| **C** | An LLM tightens the wording of each block, one block in, one block out. Code verifies each rewrite (same numbers, names, quoted strings, `must`/`never`/`always` counts) and rejects the rest. | yes |
| **D** | An LLM rewrites the whole prompt freely. A reference for the ceiling and the risk. | yes |

**The guards** apply to every approach that drops sections, whoever chose them. A section is kept if it is a global section (identity, safety, format, rules), small, named by the task, defines an identifier the remaining text uses, defines vocabulary the prompt's own examples use, or is the last section that shows how tools are defined and called. They run to a fixpoint, because keeping one section can require keeping another.

## The test cases

13 cases were written on 6 prompts. **8 are valid**, on 5 of those prompts.

| Case | Prompt | Task | What is checked | Status |
|---|---|---|---|---|
| `codex_hello` | Codex | Create hello.txt containing `hi` | Uses the patch-file format | valid |
| `gemini_newapp` | Gemini CLI | To-do web app, no questions | Names the prompt's default web stack | valid |
| `gpt41_image` | ChatGPT 4.1 | Draw a red bicycle | Calls the image tool; asks no clarifying question | valid |
| `gpt41_chart` | ChatGPT 4.1 | Line chart of monthly sales | Uses matplotlib, not seaborn (the "no colors set" check is excluded: the original fails it) | valid |
| `gpt5_bio` | ChatGPT 5 | Remember my name and units | Calls the memory tool; stores the name | valid |
| `gpt5_web` | ChatGPT 5 | Who won the last F1 Grand Prix? | Calls the web-search tool | valid |
| `v0_markdown` | V0 | Short README | Uses the markdown code-block type | valid |
| `v0_quiz` | V0 | Quiz me on React hooks | Emits the quiz component | valid |
| `bolt_chalk` | Bolt | Node project with a package | Artifact format, file order, indentation | excluded: control passed (the examples carry the format) |
| `gemini_commit` | Gemini CLI | Commit my staged changes | Runs `git status`, `git diff`, `git log -n 3` | excluded: the original fails its own check |
| `gpt5_reminder` | ChatGPT 5 | Remind me every weekday at 9am | Calendar-format schedule | excluded: the original fails its own check |
| `v0_react` | V0 | Login form component | React block type, component import | excluded: control passed |
| `v0_node` | V0 | Run some Node code | Node code-block type | excluded: control passed |

The first four valid cases were used to build and tune the approaches (rounds 1 to 3). The last four came from prompts that were not used to build anything (round 4), though the calling-convention guard was later built from one of them.

## Results by round

### Round 1: approaches A to D (2 cases)

| Case | Original (twice) | Control (key section deleted) | v1 | A | A, strict matching | B | C | D |
|---|---|---|---|---|---|---|---|---|
| `codex_hello` | pass, pass | 9% saved, **fails** | 4% held | 18% held | 34% broke | 46% **broke** | 18% held | 90% **broke** |
| `bolt_chalk` (excluded) | pass, pass | 36% saved, still passes | 13% held | 13% held | 22% held | 74% **broke** | 21% held | 91% held |

B failed because it saw only a 200-character preview. On Codex it dropped the patch-format section (its preview did not say this was the only way to create a file); on Bolt it dropped the output-format section and every example that shows the format. That is the only round-1 result that means anything for Bolt, since its control passed.

### Round 2: A, C and A+C (3 valid cases, plus Codex)

| Case | A | C | A + C |
|---|---|---|---|
| `gemini_newapp` | 19% **broke** (2 runs) | 20% held | 22% **broke** |
| `gpt41_image` | 86% **broke** (2 runs) | 8% held | 88% **broke** |
| `gpt41_chart` | 26% held | 8% held | 29% held |
| `codex_hello` | 18% held (round 1) | 18% held (round 1) | 27% held |

A dropped every tool section on the image task ("picture" shares no words with the image-tool section). C held everywhere but saves only about 8% to 20%, because its verifier rejects roughly half of the rewrites.

### Round 3: B2 on the same cases

B2 with full section text and a step plan held on all four, at 26%, 31%, 71% and 79% saved (the chart figure is before the calling-convention guard below). On the image task it wrote "generate an image using the image tool", kept that section, and dropped the other seven tool sections. Round 1's B choices for Codex, replayed through the guards, held at 37% saved.

### Round 4: a fresh test (4 valid cases)

| Case | A | B2 |
|---|---|---|
| `gpt5_bio` | 17% held | 76% **broke** (both runs) |
| `gpt5_web` | 40% held | 77% held |
| `v0_markdown` | 48% **unstable** (1 of 2 runs failed) | 54% held |
| `v0_quiz` | 43% held | 52% held |

B2's failure on the memory case: it kept every instruction about the memory tool, but dropped the other tool sections. The agent then answered in plain text and made no tool call. Those sibling sections teach how any tool is called, by example.

### The fix and its evidence

The calling-convention guard keeps one signature-bearing tool section when all were dropped, and the guards now run to a fixpoint.

| Prompt version | Saved | Runs matching the original |
|---|---:|---|
| B2, no convention guard | 76% | 0 of 2 |
| + guard, one section restored | 70% | 1 of 2 |
| + guard run to a fixpoint | 62.6% | **5 of 5** |
| `gpt41_chart` with the guard | 73% | 2 of 2 |

**Caveat:** the guard was built after seeing this failure, so the 5 of 5 is not independent proof. Those five runs (two from the diagnosis, three afterwards) were logged to the console only and are not in a results file. The one-section-restored runs and the chart runs are archived.

### Head-to-head at the same size (8 valid cases)

Every method is cut to B2's character count. Random, BM25 and vector search choose among the same sections as B2 (non-global, 400 characters or more) and keep everything else, so they inherit the parser and the global tier. None of the baselines has the guards. Selective Context (Li et al., 2023; GPT-2 self-information per sentence, task-agnostic) gets no structure at all. A failing arm was re-run once.

| Case | B2 (LLM picks, with guards) | A (word matching) | C (tightened wording) | Truncation | Random | BM25 | Vector | Selective Context |
|---|---|---|---|---|---|---|---|---|
| codex_hello | 26% held | 18% held | 18% held | 26% **broke** | 29% **broke** | 27% held | 26% held | 26% held |
| gemini_newapp | 31% held | 19% **broke** | 20% held | 31% held | 38% **broke** | 32% **broke** | 32% held | 31% held |
| gpt41_image | 71% held | 86% **broke** | 8% held | 71% **broke** | 79% **broke** | 72% **broke** | 72% **broke** | 72% **broke** |
| gpt41_chart | 73% held | 26% held | 8% held | 74% **broke** | 79% held | 73% held | 75% held | 74% held |
| gpt5_bio | 63% held | 17% held | n/a | 63% **broke** | 68% held | 63% **broke** | 63% held | 63% held |
| gpt5_web | 77% held | 40% held | n/a | 77% **broke** | 78% **broke** | 77% held | 78% held | 77% held |
| v0_markdown | 54% held | 48% **broke** | n/a | 54% held | 56% **broke** | 54% held | 54% **broke** | 54% **broke** |
| v0_quiz | 52% held | 43% held | n/a | 52% held | 53% **broke** | 52% held | 52% **broke** | 52% held |
| **Held** | **8 of 8** | 5 of 8 | 4 of 4 | 3 of 8 | 2 of 8 | 5 of 8 | 5 of 8 | 6 of 8 |
| **Mean saved** | 56% | 37% | 13% | 56% | 60% | 56% | 56% | 56% |

Sizes: each baseline is within a few percent of B2's size, and random drop ended slightly smaller (by 1 to 8 points), which makes it a marginally harder test.

## What this benchmark supports, and what it does not

**Supports**
- The redundancy-only reducer is safe and small: 2.6% of characters overall, nothing broken, instructions preserved.
- Removing the same amount of text without knowing the task usually breaks the agent (truncation and random drop held on 3 and 2 of 8).
- LLM section choice with guards kept the agent's behavior on all 8 valid cases while removing about 56% on average, and it solved the one hidden-need case that every baseline failed.

**Does not support**
- **A general claim.** Eight cases from five prompts, one or two runs each, on a single fast agent model. The checks are narrow string checks on one rule per case.
- **A clean held-out result.** The guards were built after seeing failures on cases in this set. The four "fresh" cases were fresh only until the memory failure.
- **A clear win over the best baseline.** B2 held 8 of 8 and Selective Context 6 of 8. That gap is two cases (Fisher p about 0.5). Selective Context and BM25 doing this well suggests the cases are only moderately hard: much of what the agent needs also survives elsewhere in the prompt. The cases separate B2 from truncation and random drop; they do not separate it from the other three.
- **Baselines were generous to themselves in one way and not in another.** BM25, vector search and random got our parser and global tier for free. None got the guards, and none were tuned.
- **A local-model claim.** B2 was benchmarked with a large hosted model. The showcase UI runs it with a small local model, which has been checked to work but not benchmarked.
- **LLMLingua-2 and LongLLMLingua.** Not run yet; they are the next baselines.
- **Per-task cost.** B2 needs one LLM call per task, reading the whole prompt each time. Whether that pays off depends on how many turns the agent runs.

## Reproducing

The prompts are third-party text and are not in this repository (`System-Prompts-main/`, `prompt_datasets/` and `data/enhancements-master/` are git-ignored, and `System-Prompts-main/` has no license; see the licensing note in the [README](README.md#prompt-mode)). To reproduce, place them at:

```
System-Prompts-main/*.md
prompt_datasets/piebald_claude_code/system-prompts/*.md
prompt_datasets/hf_system_prompt_library/system-prompts/json/*.json
```

```bash
# no LLM, about 1.5 seconds
python -m tests.prompt_structural_check

# behavior tests: need OPENROUTER_API_KEY in the environment or a git-ignored .env
python -m tests.behavior_pilot          # round 1: approaches A to D
python -m tests.behavior_round2         # round 2: A, C, A+C
python -m tests.behavior_round3         # round 3: B2 on the round 1-2 cases
python -m tests.behavior_round4         # round 4: fresh prompts
python -m tests.behavior_baselines      # same-size baselines (needs local models for bm25/vector/selective)
```

Models are configured in `tests/llm_config.py` (override with `SCRE_LLM_AGENT`, `SCRE_LLM_JUDGE`, `SCRE_LLM_REDUCER`). Every call is logged to `prompt_datasets/results/llm_calls.jsonl` (never the key). Free-tier models have a daily request cap and can be unavailable, so a run may need resuming; every runner skips rows it has already finished.

Result files are written under `prompt_datasets/results/` (git-ignored, since they quote third-party prompts). This report is the record of what they contained.
