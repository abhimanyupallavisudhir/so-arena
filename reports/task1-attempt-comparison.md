# Task 1 “OversightArena”: comparison of the four attempts

Evaluated 2026-09-25, at each attempt's latest pushed commit. Every attempt was installed in its own venv, its tests and demos were run, and its code was reviewed against the prompt. Suspected bugs were reproduced with scratch scripts before being listed. No attempt was modified.

| # | Agent | Extra input | Branch head | Size (src / tests) | Status |
|---|---|---|---|---|---|
| 1 | Claude Opus 5.5 (max) | — | `9c6ba55` (21 commits) | 15.2k / 1.5k lines, 2.6k lines of docs | Do stage; no PR yet, waiting for an API key for an LLM pilot |
| 2 | Claude Opus 5.5 (max) | point 8 + lab PDF | `5f7e95f` (68 commits) | 25.9k / 7.8k lines, 1.4k lines of docs | Review, PR open |
| 3 | GPT-6-astra (xhigh) | — | `76d5211` (1 commit) | 3.0k / 1.0k lines | Review, PR open |
| 4 | GPT-6-astra (xhigh) | point 8 + lab PDF | `2266343` (1 commit) | 2.8k / 0.8k lines | Review, PR open |

## Bottom line

1. **#2 is the best overall, narrowly.** It is the broadest and most robust library. It has the most tests (383 pass), and bundled real data samples for every capability-gap domain. It is the only attempt that takes on “real work” with stateful settings: forked repositories and a simulated company, with reviewers given different levels of access. It is also the most ready for real LLMs (Inspect backend, cost dry-run, cache, human-rater page).
2. **#1 is a close second.** It has the best theory (a 648-line mechanism-design write-up with propositions checked against simulation). Its verified-claims design is the most systematic (budgets, noise, forgery, who verifies), and it has the most research-shaped demos. Two bugs hit its headline features: releases leak ground truth, and the GRPO reward adapter uses the wrong stance. It also has fewer tests (102).
3. **#3 and #4 are roughly tied, well below the other two.** Each is a clean, careful, mostly correct core of about 3k lines. They share almost the same design: an async `Workflow`/`Context` plus separate evaluator sidecars. Neither is the “broad, comprehensive” library the prompt asked for. The mechanisms are about 30-line templates, realistic domains are mostly callbacks, verified claims are a data type that no protocol uses, and RL is tabular REINFORCE over fixed policies. #3 has the better theory and a ControlArena bridge that runs a real ControlArena task. #4 has more domains (chess, multi-database SQL), blinded human-review packets and a PDF review.

**No attempt has run a single real-LLM experiment.** Every demo “finding” comes from scripted or simulated agents, so it demonstrates the machinery, not the mechanisms. #1 and #2 say so; #3 and #4 say so more prominently.

## Scores (0–10)

The prompt's desires are D0–D8; general quality follows. Evidence is in the per-attempt sections below.

| | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| D0 generality (protocol + reward rule; fixtures; GT separated; domain adapters) | 8.5 | **9** | 7 | 7 |
| D1 mechanisms (debate, consultancy, monitors with CoT/probes, …) | 8 | 8 | 5 | 5 |
| D2 IC measurement (ASD, BoN trees, prompt search, steering, RL, multi-agent) | **8** | 7 | 6 | 6 |
| D3 real capability gaps + verified claims | 7 | **7.5** | 3 | 4 |
| D4 swarms (shared reward vs snitching; equilibria) | 7.5 | **8** | 5 | 5 |
| D5 beyond IC | **8** | 7 | 6 | 7 |
| D6 verdict-free release, resolve later | 6 | **8** | 7 | **8** |
| D7 mechanism-design theory | **8** | **8** | 6 | 4 |
| D8 lab PDF (#2 and #4 only) | — | 6 | — | 6 |
| Architecture / extensibility | 8.5 | **9** | 7 | 7 |
| Correctness | 6.5 | 7 | **8** | 7 |
| Code quality | 8 | 8 | 8 | 8 |
| Tests | 7 | **8** | 7 | 6 |
| Docs | **8.5** | 8 | 7 | 7.5 |
| Real-LLM readiness | 6.5 | **7** | 4 | 5 |
| Demos / findings | **5.5** | 4 | 3 | 3 |

Every attempt passed an extensibility check. From outside the library, the reviewer wrote a new mechanism, a new reward rule and a new domain, then computed ASD, without editing library code. This worked for all four; in #3 and #4 the pieces being extended are simply much thinner.

## How each attempt meets the prompt

### 1. Arbitrary mechanisms with any reward rule
- **All four** model a mechanism as a workflow plus explicit rewards for its trainable roles. An unrewarded role is a fixture, and trainable judges are allowed. #3 and #4 reject a trainable role that has no reward.
- **#2** has the cleanest formulation: game form plus composable `RewardRule`s (`Sum`, `Scaled`, `ZeroSum`, `RandomAudit`), re-applicable to logged episodes. Trainable judges and monitors are paid a proper score against inverse-probability-weighted random audits, and improper scoring rules are rejected. That makes “training the weak judge” principled rather than just possible.
- **#1** is equally general (pure reward rules over the record, re-scoring). It has the widest mechanism zoo: debate variants, consultancy, propaganda, pairwise comparison, proposer–critic, prover–verifier, monitoring with CoT/simulated/real probes and audits, peer prediction (BTS/CA/DMI), forecasting and markets, swarms.
- **#2** also has reviewed work, monitored work (output, CoT, tool calls, probe readings, diffs), teams, market making, LMSR markets, peer prediction and confessions. Its probes are lexical or embedding probes or externally attached scores, not activation hooks.
- **#3 and #4** each have debate, consultancy, monitoring and swarm templates of about 30 lines. Their monitoring does not implement chain-of-thought (CoT) monitoring or probes; #4's monitor is not exercised by any test or demo.
- **On ControlArena**, #1's `docs/controlarena.md` and #2's `docs/design.md` answer the user's question about what ControlArena's scores are, and both provide bridges in both directions. Neither bridge could be exercised here because `control-arena` isn't installed. #3's example runs a real ControlArena task offline but only imports scores. #4 only has log mappings.

### 2. Measuring incentive compatibility (IC)
- **ASD** is correct in all four: paired per item, with bootstrap intervals. #2 adds log, Brier and graded variants. #4 separates intent-only from audited ASD.
- **Best-of-N:** all four compute exact order-statistic best-of-N over a finite pool, ties included.
  - For nested (proposer → critic) trees, all four back up the *expected* value of the child subtree, not the realised draw. That is a defensible reading of “one step of RL”, but no attempt documents that it differs from literally re-sampling the procedure. A 4×4 test in #2 put the gap at about 0.16.
  - #2 goes further: each best-of-N candidate gets its own forked state, and audit and tie-break randomness is drawn per item, so best-of-N cannot select on audit luck. That is subtle and correct.
- **Prompt optimisation:**
  - **#1** gives an LLM proposer the mechanism brief and incentives, with steering (honest, deceptive or free), hidden-ground-truth constraints and held-out re-evaluation. In PSRO the proposer also sees the opponents' strategies.
  - **#2** has the same ingredients plus several search styles (OPRO, reflective, evolutionary, autoresearch-style ratchet) and the real GEPA package. Its `honesty_margin` metric is signed by the directive label, not by measured behaviour (bug below).
  - **#3 and #4** have real search loops with cohorts or strata and held-out sets. Their proposer sees only prompts and mean rewards, never transcripts, and neither supports GEPA or DSPy.
- **Multi-agent optimisation:** #1 has level-k, fictitious play and double oracle, plus empirical games with equilibria, regret and collusion gain. #2 has PSRO plus Nash, correlated and coarse-correlated equilibria, replicator dynamics and basins. #3 and #4 have finite-game regret and pure equilibria, but #4 only works on hand-written payoff tables.
- **RL:** none trains model weights.
  - #1 trains at the strategy level (REINFORCE, natural policy gradient) and has TRL GRPO and DPO adapters, but the GRPO adapter is broken.
  - #2 has an RL environment and a TRL reward function, but no training loop.
  - #3 and #4 have REINFORCE over mixtures of fixed policies.
- **2D parametric plots** (mechanism score vs true behaviour along optimisation): #1 and #2 have them for best-of-N and prompt search; #3 and #4 only for best-of-N.

### 3. Beyond toy settings; verified claims
- **#2:**
  - Bundled samples: 300 Lichess puzzles with Stockfish evaluations, a private company SQL database with 592 questions, 60 MBPP problems with hidden tests, 40 miniF2F items and GSM8K.
  - Plus QuALITY, GPQA and forecasting.
  - **Stateful “real work”:** repository tasks with hidden tests, and a simulated software company with mail, a card ledger and hidden customer state. Honest and deceptive work happens in forked branches, and reviewer access (report, diffs, queries) is an experiment axis. This speaks most directly to “protocols will stop looking like toy settings”.
  - `VerificationPolicy` makes verified claims a per-experiment switch with budgets and display rules.
- **#1** has engine-backed chess experts, a private Chinook SQL database, HumanEval+ hidden tests with mutants, Lean statement faithfulness and delayed forecasting.
  - Its `EvidencePolicy` is the most *systematic* verified-claims design: what can be checked, budget, noise, visibility, forgery and who may verify are all controlled variables.
  - Chess needs Stockfish, and `demo all` aborts without it.
  - The chess line verifier returns depth-8 evaluations, which nearly erases the capability gap the domain is meant to create.
- **#3** has a 3-row SQL scorer (agents get no SQL tool), a forecast scorer and exact match; code, Lean and chess are callbacks.
- **#4** has multi-database private SQL and a chess legality/UCI scorer.
- In both **#3 and #4**, `Claim` and receipt types exist but no protocol uses them.

### 4. Swarms
- **All four** set up shared reward vs a bounty for reporting as two comparable configurations, and all state the threshold correctly: when the bounty is below the stake, reporting pays only if others are expected to report.
- **#1** has the deepest treatment: bounty/split/leniency/misprision/forfeiture variants, an LLM-tool swarm environment, closed forms with a uniqueness proof for the mixed equilibrium, and a global-games selection argument. It also has learning dynamics: natural policy gradient escapes to honesty while REINFORCE entrenches misconduct. Two flaws:
  - The code replaces the random number of observers, $\mathrm{Bin}(n-2,o)$, with a fixed $\mathrm{round}(o(n-1))$. This misstates regimes and basins at small bounties. In the demo, at $n=3$, $o=0.8$, $b<g/3$, “all report” is not actually an equilibrium, but the code labels the case coordination with basin 0.11.
  - The global-games selection for $m\ge 3$ assumes strategic complementarity, which fails here.
- **#2** has `TeamReward` vs `Whistleblower`, covert team channels and agent-editable action logs. Its analytic threshold re-derives correctly: $p^*=1-(s/\Delta)^{1/(n-1)}$, silence risk-dominant for $n=2$ iff $s<\Delta/2$. Training demos show that the starting point selects the equilibrium.
- **#3 and #4** give the correct 2×2 analysis, but the demo game is a hand-written table. The reviewer showed #3's `Swarm` + `empirical_game` can reproduce it from real runs; #4 has no such bridge.

### 5. Beyond incentive compatibility
- **#1** has `docs/beyond_ic.md`, with each item backed by a diagnostic: signal-to-noise, spurious features, Goodhart, scaling of the capability gap, equilibrium selection, collusion (the non-zero-sum log-score debate point), validity of simulated behaviours and of ground truth, and cost.
- **#2** has §10 of its theory doc: equilibrium quality, collusion, reachability, label efficiency, cost, exploitable fixtures and failing loudly.
- **#3 and #4** each have a sensible table.

### 6. Release now, resolve later
- **#2:** SHA-256 manifest, no ground truth or arm labels in the bundle, resolve works. Minor leak: `reward_details` can carry audited labels.
- **#3 and #4:** unresolved evaluations cannot carry scores, resolution appends a new revision, and public exports were checked to be clean.
- **#1:** Merkle-committed static page, but with the default `names="auto"` the published strategy hash suffix is constant per strategy. It appears in 12 of 12 episodes, exactly those whose position was correct, so an ASD-style release reveals its verdicts. This defeats the point of the feature.

### 7. Mechanism-design theory
- **#1** covers an IC ladder, ASD as policy gradient, best-of-N with an extremal-Goodhart example, disclosure/unraveling, scoring rules, peer prediction, monitoring moral hazard, swarm propositions and learning dynamics. No false proposition was found. Errors:
  - the swarm approximation above;
  - it presents the best-of-N KL $\log n-\frac{n-1}{n}$ as exact for finite pools, where it is only an upper bound;
  - its Becker audit condition is inconsistent with the forfeiture assumption in §8.
- **#2** covers the covariance identity (ASD, best-of-N and KL-regularised RL on one path, with Goodhart as the covariance turning negative), audits as control variates, label efficiency $1/(1-\rho^2)$, implementation theory, evidence games, peer prediction and markets. Numerically checked and correct; mostly a synthesis of known results.
- **#3** has short, correct results: an alignment bound, $\frac{d}{d\beta}\mathbb E[G]=\mathrm{Cov}(G,R)$, the reporting threshold $b-c>(1-q)L$.
- **#4** is correct but prose-only.

### 8. The lab PDF (#2 and #4)
The PDF couldn't be retrieved for this review, so this is judged from what each attempt wrote.
- **#2**'s `docs/answers.md` maps the proposal's build list to modules, all self-assessed as done.
- **#4**'s `docs/lab-review.md` maps PDF sections to features and adds useful caveats: the Brier convention, “known by construction” vs manipulation checks, and graded-ASD normalisation.

Neither changed the design much because of the PDF. #4's review is more critical; #2's stateful “firm” domain plausibly came from the PDF.

## Verified bugs (most important)

| # | Severity | Bug |
|---|---|---|
| 1 | High | TRL GRPO `reward_function` scores every completion as position A, ignoring its stance: the A-argument got −0.105 and the B-argument −2.30 (`elicitation/rl.py:256-289`). |
| 1 | Med-high | Release strategy hashes reveal ground truth (`release/release.py:52`). |
| 1 | High (by design, undocumented risk) | Model code runs unsandboxed. It could read the parent process's environment and the cached mutant file that identifies the wrong answer. |
| 1 | Med-high | `OptimizationTrace.best` drops candidates whose reward is exactly 0: with rewards [-1, 0] it picks -1. This feeds prompt search's final pick, held-out re-evaluation and PSRO best responses. Found by another reviewer, confirmed here ([repro](repro/a1_best.py)). |
| 1 | Med-high | The hidden-test grader passes code that prints `PASSED` and exits before `check()` runs. This grader is the ground truth for agent-written code. Found by another reviewer, confirmed here ([repro](repro/a1_code.py)). |
| 1 | Medium | Judge's own SQL claims are auto-verified under default `EvidencePolicy`. Verification noise has no effect on successful SQL and code results. `LeanVerifier` accepts axioms. |
| 2 | High | Agents with a shell can read hidden ground truth: the firm's `hidden.json` and the bundled MBPP hidden tests. State temp dirs are never cleaned up. |
| 2 | Med-high | “Verified” laundering: a claim with no expected value is marked verified if it merely executes (e.g. `SELECT 'Option A is correct'`). Forged `<verified>` markers slip through via homoglyphs or HTML entities, and reach the monitor unescaped. |
| 2 | Med-high | The game-tree solver has no information sets: node keys are the full decision history and only simultaneous moves are coupled. A blind coin guesser gets best-of-2 success 1.0, where 0.5 is the maximum. Found by another reviewer, confirmed here ([repro](repro/a2_coin.py)). |
| 2 | Medium | Tree summaries renormalise over labelled leaves and report no coverage. Bo(64) reports correct = 1.0 even though almost all selection mass is on an unlabelled candidate. Found by another reviewer, confirmed here ([repro](repro/a2_missing.py)). |
| 2 | Medium | The PSRO n-player “Nash” is an unchecked replicator iterate, reaching NashConv 2.0 on 3-player pennies. `honesty_margin` is signed by directive label. With a fixed seed, audits are the same draw on every RL step. |
| 3 | Medium | `MixturePolicy` redraws its component on every call, switching mid-episode. `InspectPolicy` uses bare `json.loads`, so fenced JSON crashes the episode. |
| 4 | Medium | Joint prompt search re-evaluates the incumbent under the same seed, so a stochastic policy trips the `Store` “conflicting immutable record” check. The proposer never sees rejected candidates. `InspectPolicy` has the same bare `json.loads`. |

## Tests and demos run here

| # | pytest | Demos |
|---|---|---|
| 1 | 99 passed, 3 skipped, of 102 collected (the report said “~110”) | `demo all` aborts at chess without Stockfish; the other demos all run and reproduce the documented numbers; mock-LLM examples run |
| 2 | 383 passed, 5 skipped (the report said 389) | `demo all` builds all 4 reports; `run`, `estimate` (pilot set ≈ \$0.20), release/verify and an Inspect mock-LLM ASD run all work |
| 3 | 67 passed, 2 skipped (71 with extras) | Demo, examples and resolve/export/report CLI all run; `--plots` needs matplotlib |
| 4 | 45 passed, 4 skipped (49 with extras) | Demo (109 runs), resolve, report and examples all run; `--plots` needs matplotlib |

Skips are missing optional packages or binaries (control-arena, Stockfish, inspect_ai, chess, network), not failures.

## If keeping one

Take **#2** as the base, then:
1. Port from #1:
   - `docs/theory.md`, correcting the swarm $\mathrm{Bin}(n-2,o)$ issue and the KL wording;
   - `beyond_ic.md` and the ControlArena explainer;
   - `EvidencePolicy`'s noise, forgery and verifier-identity axes;
   - the extra mechanisms (prover–verifier, pairwise, BTS/CA/DMI);
   - the monitoring-obfuscation and “lies move beyond the verification budget” demos.
2. Fix #2's shell leak (sandbox, or keep hidden state outside the agent's filesystem), verified-claim laundering, the n-player Nash check and the `honesty_margin` sign.
3. Run the real-LLM pilot. Both Claude attempts need only an OpenRouter, OpenAI or Anthropic key; #2 prices its pilot set at about \$0.20.

#3 and #4 add nothing the others lack except #4's PDF caveats and #3's genuinely-running ControlArena example. Both are worth a quick look, but not worth merging.

A second-pass bug hunt with about 110 more findings is in [bug-hunt.md](bug-hunt.md).
