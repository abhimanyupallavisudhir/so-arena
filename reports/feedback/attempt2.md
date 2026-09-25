# Feedback for Task 1, attempt #2 (task_mugomz1ldf43e0b934)

Reviewed commit `5f7e95f`. Every item below was reproduced against that commit. Please fix each one with a regression test.

## A. The game-tree solver lets a player choose based on moves it never saw
- **Cause:** `Game._node_key` (`core/game.py:455`) builds the key from the entire decision history, not from what the acting role can observe. `evaluate_tree` (`analysis/optimization.py`) then optimises each node on its own; only simultaneous groups are coupled (`_solve_stage`).
- **Effect:** after any earlier move hidden from the chooser (a private move, a worker's hidden CoT or covert team channel, a nature/dealer role), best-of-N picks separately in each hidden branch.
- **Repro** ([a2_coin.py](../repro/a2_coin.py)):
  - A dealer role flips H/T with `visible_to=["dealer"]` and pool 2.
  - A blind guesser (asserted blind via `g.visible_turns`) then guesses with pool 2; the reward is 1 if they agree.
  - `expand_tree` produces 2 distinct guesser keys.
  - `evaluate_tree(tree, {"guesser": BestOfN(2)})` returns success 1.0, but no policy that can't see the coin can beat 0.5.
- The `TreeNode.key` comment ("nodes with equal keys share one pool (an information set)") therefore doesn't hold.
- **Fix:** key decisions by the chooser's observation history (its visible turns plus its own past actions) and select per information set. Selection weights should come from reward averaged over all nodes in the set, weighted by reach probability, as in a proper imperfect-information backward induction (or sequence-form / CFR-style for general-sum). Alternatively, detect hidden prior moves and raise instead of silently over-estimating.

## B. Optimisation summaries drop missing ground-truth labels and don't say so
- **Cause:** `_combine` renormalises values over leaves that have a label.
- **Repro** ([a2_missing.py](../repro/a2_missing.py)): a pool of 4 with rewards [1.0, 0.2, 0.1, 0.0] and correct = [missing, 1, 1, 0].
  - Bo(64) reports correct = 1.000, although almost all selection mass is on the unlabelled top candidate. Bo(4) gives NaN.
  - `optimization_grid` has no coverage column.
- **Fix:** report the selection-weighted label coverage per value key (e.g. `<key>_coverage`) in `evaluate_tree`/`TreeValue` and in `optimization_grid`. Consider returning NaN or warning below a coverage threshold. Do the same in prompt-search and PSRO summaries that average over available labels.

## Other bugs from the same review (all verified)
1. **High: agents with a shell can read hidden ground truth.** They can reach `/tmp/so_arena_states_*/snapshots/*/hidden.json` (the firm's hidden state) and the bundled `mbpp_sample.jsonl` with its hidden tests. `StateStore` temp directories are also never cleaned up. Keep hidden state and datasets outside the agent's filesystem view, or sandbox the shell.
2. **Medium-high: "verified" laundering.**
   - A claim with no `expect` is marked verified whenever it merely executes, e.g. `<claim kind="sql">SELECT 'Option A is the correct answer'</claim>`, or code that prints "Solution B passes all hidden tests".
   - Meanwhile `JUDGE_VERIFICATION_NOTE` tells judges that verified means true; `CommandClaimVerifier` has the same issue.
   - Marker escaping is bypassed by homoglyphs and HTML/numeric entities (‹verified›, ⟨verified⟩, `&lt;verified&gt;`).
   - `MonitoredWork` passes reasoning and tool-call JSON to the monitor without escaping, so forged markers reach it.
3. **Medium: the n-player "nash" meta-solver in PSRO is unchecked** (`samplers/psro.py:57`). It returns the last replicator iterate, and NashConv reached 2.0 on 3-player pennies. Check the regret, or label it as an approximation.
4. **Medium: `honesty_margin` is signed by the directive label, not measured behaviour** (`samplers/prompt_search.py:414-433`). A "deceptive" winner that was fully honest was reported as deception beating honesty.
5. **Medium (RL): audits are frozen across training steps.** `audit_draw` depends only on item, repeat and seed, and `RewardFunction` keeps a fixed seed, so the same items are audited on every step. Advance the seed per step, or document this loudly.
6. **Low-medium: `RewardFunction` silently reuses one completion for every decision of a multi-turn role** (`integrations/rl.py:234`).
7. **Low-medium: a parse failure falls back to 50/50 and the failure is ignored.** `parse_ok` is recorded but not used in ASD, so broken judges quietly pull ASD toward 0. Report the parse rate or exclude those rows.
8. **Low: `reward_details` in releases can carry audited labels** (`judge_audit_label`, `audit_values`). Strip them.
