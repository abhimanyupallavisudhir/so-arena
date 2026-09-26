# What was brought over from attempt #1, and what else was fixed

2026-09-26. Master is attempt #2 (`so_arena`, merged as PR #3). Attempt #1 (`oversight_arena`, task #1 attempt 1, branch head `db90587` after its own fix rounds) was compared against it feature by feature. Where #1 was better, the idea was re-expressed in master's abstractions: Mechanism + RewardRule, `Game`/`g.chance`, `VerificationPolicy`/`Verifier`, domains, samplers, games, theory. #1's code was not pasted in. #1's known bugs were not carried over (see [bug-hunt.md](bug-hunt.md) and [fix-review.md](fix-review.md)).

Tests: **738 pass, 0 skipped** with the optional control-arena 19.0.0 and Stockfish 17.1 installed. Without them it is 735 passed and 3 skipped. Lint (`ruff --select F,E9,W605`) is clean.

## Brought over

| Area | What master gained | Notes |
|---|---|---|
| **Monitoring** | `MonitoredWork` options (missing scores fail closed, `defer=True` to a trusted role, `forfeit=`); `MonitoredReward` (re-scorable); `SimulatedProbe` with a fixed AUROC and draws from `g.chance`; a synthetic monitoring domain; a ground-truth access ledger (`Episode.gt_access`); `theory/monitoring.py`; theory §6.1 with Proposition 4; `demo_monitoring` | **Fixes a fail-open bug in master:** an unparseable monitor answer used to score 0.5 and was not flagged. Measured thresholds match theory within CIs (keyword monitor turns covert at λ=0.0587 vs 0.0588 in theory; a probe deters at 0.467 vs 0.472). |
| **Verified claims** | `VerificationPolicy(noise=, costs=, budget=, show_to=)`: noise with forged-but-realistic outputs for every verifier, drawn per item/role/claim so best-of-N can't select on it, with the true status kept for ground truth; cost budgets; control over who sees verdicts. Verifiers that can't forge believably fail closed. | HiddenBits disclosure domain with exact Bayesian judges (naive, credulous, sceptical); `theory/disclosure.py`; theory §5 with Proposition 3; `demo_hiddenbits` and `demo_bon_budget` (best-of-N moves a liar's lies beyond the verification budget: 0.82 → 0.02 checked vs 1.67 → 3.66 unchecked). |
| **Swarms** | The whistleblower model with partial observation (other reporters $\mathrm{Bin}(n-1, o\,p)$, threshold bounty $b_R$, regimes, a 2-D mean field, basins); `Team(observe_prob=, self_reports=)`; `Whistleblower(leniency=, misprision=)`; `SyntheticTeam` gain/damage/opportunity; natural policy gradient, and `StrategyGradient` on sampled episodes | **Replaces master's weaker everyone-observes model**, where "all report" could look stable below $b_R$. Theory §8 now has Propositions 6–7. The demo shows natural gradient deterring in 6/6 runs vs REINFORCE in 0/6 from the same start. |
| **Mechanisms** | `Debate(n_debaters=k, cross_examination=True)`; `Comparison` + `PreferenceScore` (the RLHF / reward-model signal); `ProverVerifier` + `ProverReward` (mode from `g.chance`, misaligned penalty below every aligned score); peer prediction CA and DMI (DMI with a fixed normaliser, #1's bug fixed); a `Forecast` mechanism with a synthetic forecasting world; level-k PSRO; `demo_release` | Proposition 5 (an immediate proxy is proper for the judge, not the outcome): the released ranking puts the extremizing forecaster first, and resolution puts it last. |
| **Analysis and training data** | `analysis/diagnostics.py` (reward SNR, length/slot/label biases at fixed ground truth, ECE, compliance, cost); `metrics.gt_regret`; DPO `preference_pairs` (same-context pairs only); `ParamSearch` over programmatic agents' parameters | The diagnostics appear in HTML reports when ground truth is known. |
| **Registry** | `so_arena.registry`: any registered reward rule, verifier, scorer or mechanism can be built from `{type: ...}` specs, recursively, with errors checked before anything runs; `so-arena list rewards|verifiers|scorers` | Replaces the four reward rules hard-coded in `spec.py`. The existing configs keep their config hashes. A test fails if a new component isn't registered. |
| **Theory and docs** | Propositions 1–2 (ASD as the policy gradient; extremal Goodhart under best-of-n); the revelation principle; log-score debate is not zero-sum; stochastic stability; the meta-solver table; how oversight differs from classical mechanism design; open problems; `docs/beyond_ic.md`; `docs/controlarena.md`; `docs/reference.md` (generated) | Propositions are numbered 1–7 in document order. |
| **Chess** | `engine_advocate` (honest or cherry-picked `chess_line` claims) and `engine_judge` (minimax over the lines shown, at a shallow depth that sets its strength); `ChessDomain(trap_depth=)`; `demo_chess` | Stockfish-backed. #1's depth-8 line verifier, which erased the capability gap, was not brought over. Depth-2 judge on trapped positions: 0.00 alone, 0.94 with honest lines, 0.58 with cherry-picked lines in consultancy and 0.78 in debate. |
| **Sandbox** | A Landlock + seccomp backend for containers without user namespaces, and as a second layer under bubblewrap/unshare | It refuses to run when isolation can't be enforced, unlike #1, which silently fell back to an audit hook. Tested on this kernel (Landlock ABI 2). |
| **Releases** | `release(sealed=True)`: salted per-item commitments with Merkle inclusion proofs, `reveal()` item by item, and third-party `inclusion_proof()` checks | Master's leak-proofing is unchanged: sealed contents equal what an unsealed release would publish. |

## Deliberately not brought over (master is equal or better)

- **Judging mechanisms:** master's `DirectJudge`, `Propaganda` and `Consultancy` are equal.
- **Proposer–critic:** master's `ReviewedWork` is better (critique, rebuttal and stateful dossiers).
- **SimOps swarm environment:** master's repository team domain is better, and SimOps still had an open evasion.
- **Datasets:** master's QuALITY, MCQ, SQL, code, Lean and GSM8K are equal or better.
- **Prompt optimisers:** master's prompt search and GEPA are better.
- **RL environment and GRPO adapter:** master's `MechanismEnv` and TRL adapter are better.
- **Best-of-N trees and game solvers:** master's are better; it has information sets, CE/CCE and coalition deviations.
- **Human-judge web page:** master's `rating.py` is better.
- **Docs:** #1's `concepts.md` and `cookbook.md` are covered by `design.md`, `experiments.md` and `extending.md`.
- **Also not done:** `ControlArenaDomain` (ControlArena settings as a domain). Master's bridge covers logs → episodes and mechanisms as monitors.

## Other issues fixed in the repo

- **Chess `eval_claim` items could be answered without reading the position.** A blind two-feature rule scored 0.83, worse than the 0.7 documented. Positions are now chosen for the whole set (`balanced_eval_claims`, a small integer program) so that yes and no balance within every combination of check, material, mobility, last capture and side to move. Every blind lookup rule now scores 1/2, and a 20-feature logistic model scores 0.58. All 298 of 300 puzzles are kept.
- **Chess `which_move`:** `ChessDomain(balanced=True)` removes the remaining "pick the check" tell (0.587 → exactly 1/2, keeping 231 of 300 items). The default keeps all items, and the residual tell is documented. `blind_baselines()` reports both kinds of item.
- **LaTeX in docstrings:** several docstrings had LaTeX escapes that Python read as control characters (`\frac` → form feed) or invalid escapes. A test now guards all docstrings.
- **Release demo colours:** the charts showed a distortion's gain in the "good" colour. `asd_bars(positive_is_good=)` fixes this.
- **Figures script:** `scripts/make_figures.py` now covers every demo and takes demo names.
- **Unused code:** unused imports and locals are removed.
- **Demo reports regenerated** from the final code (`scripts/make_figures.py`). They reproduce the numbers in each package's own report, except the verified-debate row of the ASD demo. `VerificationPolicy` gained fields, so its config hash and thus the scripted agents' random draws changed: ASD 3.07 (CI 2.65–3.53), was 2.96 (2.53–3.35). The ASD report also gains the new diagnostics table. `demo_monitoring`'s defaults now match its published run (300 tasks × 30 seeds).
- **ControlArena bridge:** exercised for the first time in these reviews, against control-arena 19.0.0. All its tests pass.

All round-3 review items for #2 (information sets keyed on hidden state and on turn indices, run directories visible to the sandbox, the forgeable tests verifier, `expect=` laundering, SQL bullet lists, collective-liability escape, frozen `MechanismEnv` audits, parse status in plain ASD) were already fixed in master before this work. Their repro scripts were re-run and confirm it.

## Still open

- **No real-LLM results yet.** All demos use scripted or simulated agents. The pilot scripts (`scripts/pilots.py`, about \$0.20) need an OpenRouter, OpenAI or Anthropic key.
- **Verifiers without realistic forgeries:** chess, Lean, repository-test and visible-test verifiers can't produce believable fakes, so verification noise on them fails the episode rather than faking a result.
- **Which claims get checked:** verification noise and budgets apply to the first k claims. A random subset of checked claims would stop an optimiser learning where lies are safe; the theory describes it, but it isn't implemented.
- **Landlock backend limits:** when Landlock runs on its own, `/tmp` and the home directory are invisible rather than fresh scratch space. Callers already set `HOME` and `TMPDIR` to the working directory.
