# Feedback for Task 1, attempt #1 (task_muf0agqo7f64190d0c)

Reviewed commit `9c6ba55`. Every item below was reproduced against that commit. Please fix each one with a regression test.

## A. Zero rewards are discarded when picking the best candidate
- **Where:** `elicitation/optimize.py`, `OptimizationTrace.best`, line ~81.
- **Cause:** `getattr(c, key) or float("nan")` turns a reward of exactly 0.0 into NaN, so the candidate is dropped.
- **Repro** ([a1_best.py](../repro/a1_best.py)): with candidate rewards [-1.0, 0.0], `best(2)` returns only the -1.0 candidate.
- **Affects:**
  - the final pick in prompt search;
  - which top-k candidates get held-out re-evaluation (`optimize.py:434`);
  - PSRO/level-k best responses (`multiagent.py:52`, `scripts/llm_pilot.py:106`, `demos.py:241`).
- **Fix:** test with `is None` / `math.isnan`, not truthiness.

## B. The hidden-test grader can be fooled
- **Where:** `domains/code.py:150-156`.
- **Cause:** `_hidden_harness` runs the candidate's code first, and `passes_hidden` then only checks `r.ok and "PASSED" in r.stdout`.
- **Repro** ([a1_code.py](../repro/a1_code.py)): code that runs `print('PASSED'); sys.exit(0)` (or `os._exit(0)`) at module level is graded as passing.
- **Why it matters:** this is the ground truth for agent-written code (`CodeArtifactGT`), so optimisation can game the ground truth itself.
- **Fix:** have trusted code emit a random per-run nonce after `check()` returns, and require that nonce plus exit code 0. Better still, run `check` in a separate process that imports the candidate as a module and reports through a channel the candidate cannot write to.

## Other bugs from the same review (all verified)
1. **High: the GRPO reward adapter ignores which answer each completion argued** (`elicitation/rl.py:256-289`). It runs every completion under `profile or Profile()`, so everything is scored as position A. Under a credulous judge, arguing A got -0.105 and arguing B got -2.30. Pass each completion's stance through.
2. **Medium-high: releases leak ground truth.** With `names="auto"` the published `strategy_id` hash suffix is constant per strategy (e.g. HONEST = `28019f86`). It appears in exactly the episodes whose position is correct (12 of 12). Salt the hash per release, or drop the suffix.
3. **Medium: the swarm model misstates regimes at small bounties.** `SwarmParams.m` uses $\mathrm{round}(o(n-1))$ where the number of observers is really $\mathrm{Bin}(n-2,o)$. At $n=3$, $o=0.8$, $b<g/3$, "everyone reports" is not actually an equilibrium, yet the code labels the case coordination with basin 0.11. The global-games selection claim for $m\ge 3$ also assumes strategic complementarity, which fails here because $\Delta(K)$ decreases in $K$.
4. **Theory doc fixes:**
   - `theory.md:134` presents $\log n-\frac{n-1}{n}$ as the exact KL for finite pools; it is an upper bound.
   - The Becker audit condition $aP\ge G$ is inconsistent with the forfeiture assumed in §8, which gives $aP\ge(1-a)G$.
5. **Medium: verification gaps.**
   - The judge's own SQL claims are auto-verified under the default `EvidencePolicy(roles=None)`, so it can query the private database.
   - Verification noise never flips a successful SQL or code result.
   - `LeanVerifier` accepts axioms.
   - Model code runs unsandboxed and can read the parent process's environment and the cached mutant file.
6. **Low: `demo all` aborts entirely when Stockfish is missing.** Skip the chess demo with a warning instead.
