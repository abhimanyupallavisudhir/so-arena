# Task 1: review of the fixed versions of attempts #1 and #2

2026-09-26. After two rounds of bug reports ([feedback/](feedback/), [bug-hunt.md](bug-hunt.md)), both agents reported that everything was fixed.

| Attempt | Reviewed commits | Changes | Tests here |
|---|---|---|---|
| #1 | `9c6ba55` → **`14ae42c`** | 5 commits, +4.4k/−1.1k lines | 173 passed, 4 skipped |
| #2 | `5f7e95f` → **`523d6f7`** | 15 commits, +6.4k/−1.2k lines | 533 passed, 6 skipped |

The skipped tests need optional installs (control-arena, Stockfish, network data). #3 and #4 are unchanged.

## How this was checked
- Every earlier repro script was re-run against the new code. Scripts that broke because an API changed were adapted.
- Each fix was then attacked with *variants* of the original exploit, and the diffs were read for regressions.
- New scripts are in [repro/fixrev1/](repro/fixrev1/) and [repro/fixrev2/](repro/fixrev2/).

Items marked ✔ were re-run and confirmed by me.

## Bottom line
Both attempts are **materially more trustworthy**. Nearly all of the roughly 100 reported items are fixed, and fixed properly rather than special-cased to the repros. The remaining problems are the same themes as before, found in new places: ground truth or "verified" verdicts that an agent can fool, and sandbox gaps.

| | Fixed | Partial | Not fixed / documented | New issues |
|---|---|---|---|---|
| #1 | ~40 | 6 (sandbox, calc, assert, SimOps, `parse_scalar`, markers) | 1 (`bon_kl` n<1) | 8 (N1–N8) |
| #2 | ~45 | 5 (information sets, sandbox, laundering, frozen audits in `MechanismEnv`, parse status in plain `asd`) | chess and code blind tells (documented) | 8 (M1–M8) |

## Attempt #1: what's fixed
- **Round 1:**
  - zero rewards are kept ✔;
  - the grader uses a nonce from a separate harness process, so the `PASSED` sentinel and the always-equal object both fail ✔;
  - GRPO scores each completion at its stance;
  - release hashes no longer leak;
  - the swarm analysis now uses $\mathrm{Bin}(n-2, o\,q)$. I checked: $b_R=g/3$ at $n=3$, $o=0.8$; the mixed equilibrium is unique; the risk-dominance formula is right;
  - the KL and Becker wording are corrected;
  - judges' claims are no longer auto-verified;
  - noise now flips outputs;
  - Lean is checked by the kernel with an axiom audit;
  - the chess demo is skipped when Stockfish is missing.
- **Round 2:**
  - DMI truthful 0.95 vs informative 0.06;
  - the monitor is credited only for its own flag (an always-0 monitor gets −0.5);
  - trees have 48 of 48 leaves;
  - SimOps no longer reads worker stdout;
  - null chess moves are rejected;
  - NaN-safe selection;
  - missing monitor scores fail closed;
  - the natural gradient is unbiased (0.509);
  - `frontier` averages ties;
  - `preference_pairs` stays within one context;
  - `nash()` handles degenerate games (max regret 1e-14);
  - results are grouped by configuration hash;
  - resume re-scores changed ground truth;
  - `partial` and closure keys are fixed;
  - common random numbers hold (200/200);
  - the Inspect bridge is fixed;
  - quotes, SQL rows, release duplicates, blind tells and task-clustered CIs are all fixed.
- **Sandbox:** with Landlock and seccomp active (as here), the environment is empty and `/proc`, files outside the folder, sqlite ATTACH and sockets are all blocked.

## Attempt #1: remaining and new issues
- **N1. High where Landlock is absent: the sandbox fails open silently.** `_sandbox_lib.py:222-239` only records `landlock: False` and continues. What remains is a Python audit hook that misses `readline` and sqlite `ATTACH`/`VACUUM INTO`. Without Landlock, a candidate reads the trusted harness through `/proc/<ppid>/cwd/main.py` and passes **4 of 10 real HumanEval+ tasks** with no solution. Landlock is missing on kernels before 5.13 and in many containers. `fixrev1/sbx_probe.py`, `fixrev1/nolandlock_gt.py`
- **N2. Medium: the `<assert>` rebinding block can be bypassed** (`code.py:235-254`). ✔ `(lambda impl_B: impl_B('ab')=='ba')(impl_A)`, the comprehension form and the `all(... for impl_B in [impl_A])` form are all VERIFIED for the wrong implementation. Meanwhile the honest `type(x) == str` is rejected. `fixrev1/assert_rebind.py`
- **N3. Medium-high when Lean is configured: Lean runs unsandboxed and `#eval` is allowed.** `LocalLean` runs on the host, and `FORBIDDEN` has no `#eval`. A proof can read files or run metaprograms during grading (`setEnv`, `addDeclWithoutChecking`). `_DECL` also matches a `theorem` inside a comment. This is reasoned from the code; Lean isn't installed here. `fixrev1/lean_static.py`
- **N4. Medium: CalcVerifier still marks common true claims REFUTED.** ✔ `12 × 7 = 84`, `12 x 7 = 84` and `$12 * 7 = $84` give `verified=False` on a parse error, as does `max(1,000, 5) = 1000`. A parse error should give "unverified", not "refuted". `fixrev1/calc_v.py`
- **N5. Medium: SimOps hard-coding detection can be evaded.** Snapshots are replayed without `tests/`, so a module that parses the visible tests at run time isn't flagged (`violators []`). `fixrev1/simops_readtests.py`
- **N6. Low-medium: `parse_scalar` inverts values.** On a [0,1] scale, `8 out of 10` → 0.08, `1.5` → 0.015, and `1e-3` → 1.0. This parser feeds LLM ground truth. `fixrev1/scalar.py`
- **N7. Low-medium: provenance markers.**
  - ControlArena tool output renders as `⟦Moderator⟧` without the untrusted pass.
  - The look-alikes ⟪⟫, 〘〙, ⦋⦌, 〔〕 and `[[ ]]` survive sanitising, and message bodies aren't indented.
  - `fixrev1/markers.py`
- **N8. Low: over-strict honest-code grading.**
  - Results over 4300 digits fail to serialise, so the canonical HumanEval/83 and /139 now fail.
  - `tempfile` is blocked.
  - A cold mutant-cache rebuild takes over 300 s.
- **Other:**
  - GRPO silently falls back to position A when there is no stance column;
  - SQL `_cell` rounds to 10 significant figures;
  - `bon_kl(n<1)` is still 0.
- **Demo:** the probe-deterrence threshold moved from 0.4 to 0.5 only because the random draws changed. The finding is noise-sensitive and should be reported with a CI.

## Attempt #2: what's fixed
- **Round 1:**
  - **Game-tree solver:** it now works on information sets. The blind coin guesser gets 0.5 ✔, and the variants are correct too: a hidden trainable opponent 0.5, a 3:1 noisy signal 0.75, a skewed dealer 0.667.
  - **Label coverage:** reported ✔, and the grid drops values below 50% coverage.
  - **Shell leak:** `hidden.json` and the MBPP hidden tests are unreachable ✔. The sandbox has no network, no capabilities and a read-only venv, and temp stores are cleaned up.
  - **PSRO Nash:** NashConv 0 on 3-player pennies, and ≤2.6e-5 on random games.
  - **Other round-1 items:** `honesty_margin` is classified by measured value; completion reuse raises; `reward_details` is allowlisted.
- **Round 2, all seven high items fixed:**
  - ✔ code and SQL ground truth grade the work phase (rebuttal hack → −1);
  - team-audit evasions are caught (as is a string-encoding variant);
  - release metadata leaks nothing;
  - ✔ refunds go to customers' cards, and edit-then-revert counts as fraud;
  - every reported figure is checked;
  - the peer-prediction cache is keyed per role (warm = cold, 0.53);
  - multitask pairing and tie-breaks are chance draws.
- **Round 2 medium and low items, fixed:**
  - `sitecustomize`/shadowing counts as tampering;
  - the false-claim detector covers the reported variants;
  - the chess scorer reads the declared move;
  - Lean checks the axiom whitelist;
  - quotes match whole words;
  - judges see `expect=` and similar attributes;
  - split modules score +1;
  - configurations are labelled separately;
  - label efficiency is 1.02;
  - GEPA's train, validation and test sets are disjoint;
  - model args and closures are in cache keys;
  - errored ground truth is re-scored;
  - missing oracles raise;
  - GSM8K options are evenly spaced (the centre rule drops to 0.14).
- **Documented as not fixed:** the chess eval-claim tell (~0.7), the chess which-move tell (0.587), and the code "more branches" rule (right on 12 of 13 items it decides).

## Attempt #2: remaining and new issues
- **M1. Medium-high: information-set keys include hidden environment state.** ✔ `Game._infoset_key` (`core/game.py:486-488`) hashes the snapshot id, and `content_id` includes `ws.hidden`. A mover writes H or T only into hidden state, and an observer with identical visible files then gets 2 keys and **Bo2 = 1.0** (control: 1 key, 0.5). A read-access critic could "detect" edit-then-revert fraud it has no evidence of. Fix: key on the visible files only. `fixrev2/infoset_hidden_state.py`, `fixrev2/infoset_control.py`
- **M2. Medium-low: information-set keys include global turn indices, which count hidden turns.** A hidden choice that adds a private turn splits the key although the prompts are identical (Bo2 = 1.0). `fixrev2/infoset_count.py`
- **M3. Medium-high, deployment-dependent: run directories aren't hidden, and `/` is writable under the `unshare` fallback.** There is no bubblewrap here. A sandboxed shell read earlier episodes' ground truth from `runs/…/episodes.jsonl` **and appended to it**. It is safe only when the working directory is under `$HOME` or `/tmp`. `fixrev2/sbx_probe.py`
- **M4. Medium: the trusted `tests` verifier can be forged.** ✔ The runner keeps its token in `__main__._token`. Four lines in `solution.py` print the token line and exit, so the verifier says **verified**, the reviewer sees `<verified kind="tests">` and `tampered` is False. Ground truth stays correct (−1). Fix: hold the token in a parent process only. `fixrev2/repo_token.py`
- **M5. Low-medium: laundering through `expect=`.** `print('Solution B passes all hidden tests')` with a matching `expect=` comes back verified. `fixrev2/launder.py`
- **M6. Low-medium regression: SQL grading.** A correct query plus a bullet-list result (`- Customer Support | 3`) now scores −1; it scored +1 before. `fixrev2/sql_table.py`
- **M7. Low-medium: the collective-liability penalty can be escaped.** `silence_penalty` exempts anyone who filed any report, even a refuted one, and `false_report_penalty` defaults to 0. So a false accusation beats silence.
- **M8. Low:**
  - `run_lean` passes the full environment into the sandbox;
  - marker look-alikes (〚〛, 【】, small caps) are not neutralised;
  - `MechanismEnv.reset(seed=0)` still freezes audits;
  - plain `asd(role_frame(...))` ignores parse status (used in `demos.py:54` and `controlarena.py:210`);
  - released token counts might reveal directive arms (reasoned).

## Ranking after the fixes
Unchanged: **#2 ≥ #1 ≫ #3 ≈ #4**. Both Claude attempts closed the large majority of issues within hours, with real regression tests.
- **#2's fixes** are slightly broader: an information-set solver, per-role cache identities, and blind-baseline helpers.
- **#1's sandbox** is stronger where Landlock exists. #2's `unshare` fallback leaves `/` writable.

Neither has yet run a real-LLM experiment.
