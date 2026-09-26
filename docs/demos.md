# Demos

Seven built-in experiments run without API keys, using exact programmatic agents (Bayesian judges,
engine-backed chess players, simulated forecasters and workers):

```bash
oversight-arena demo all --out runs/demos          # or one of: hiddenbits chess bon optimize swarm monitoring release
python scripts/build_gallery.py                    # rebuild docs/gallery/ and docs/figures/
```

Each demo writes an HTML report with tables, figures and a transcript browser. The committed
gallery is [docs/gallery/index.html](gallery/index.html). The demos also serve as templates for LLM
experiments: swap the agents for `oa.llm_agents(...)`.

## Verified claims (`hiddenbits`)

**Setup.**
- Experts see 15 hidden bits; the question is whether at least half are 1.
- Advocates reveal bits as claims that trusted code can check, lying when favourable true bits run
  out.
- Judges are exact Bayesian: *rational* (ignore unverified claims) or *credulous* (trust 0.8).

**Finding.**
- Debate's ASD grows steeply with the verification budget: from about 0.12 without verification to
  about 1.3 at 6+ checked claims for the credulous judge.
- Consultancy and propaganda stay near zero.
- With a rational judge, nothing works without verification (cheap talk carries no information),
  and only debate unlocks at larger budgets.

Both effects match the exact disclosure theory (`theory.disclosure`, theory.md Proposition 3).
Competition substitutes for a sceptical judge.

![ASD vs verification budget](figures/hiddenbits_asd_budget.png)

## Chess: a real capability gap (`chess`)

**Setup.**
- Tasks are Lichess puzzle positions where the weak judge's own depth-2 search prefers the wrong
  one of two moves; ground truth is Stockfish at depth 14.
- Engine-backed advocates argue with chess lines that trusted code checks. *Honest* lines show
  best play; *cherry-picked* lines assume poor replies.

**Finding.**
- Alone, the depth-2 judge is wrong on every position (0% accuracy); arguing for the true move is
  *penalised* (ASD −0.98).
- With honest verified lines, consultancy and debate reach 87% accuracy (90–93% for depth-1 and
  depth-3 judges).
- With cherry-picked lines they fall to 50–57%.
- In the argument-style game between the debaters, both show honest lines in equilibrium; the
  judge is 87% accurate.

![Chess, honest advocates](figures/chess_accuracy_honest.png)

## Best-of-N (`bon`)

**Setup.** 32 base-policy samples per task, with random claim counts, lie rates and side errors.
Selection is exact best-of-n by the mechanism's reward (KL axis available in `bon_curve`).

**Finding.**
- For a consultant with verified claims, selection raises the honesty of claims from 0.85 to 0.99.
- In debate, selecting the (assigned) liar's best samples lowers honesty. Selected samples almost
  never lie in verified claims (0.01 per episode, against 0.5 for the rest) but lie beyond the
  verification budget four times as often (2.75 against 0.67), where the credulous judge still
  believes them. Optimisation moves deception to where verification does not reach.

![Best-of-N](figures/bon_honesty.png)

## Strategy search (`optimize`)

**Setup.** Evolutionary search over advocate strategies (claim counts, lie rates, markup) under
three steerings: honest only; anything while arguing the truth; anything while arguing a
falsehood. Every search is re-evaluated on held-out tasks.

**Finding.**
- Debate (budget 4) keeps a frontier gap of 0.25 between the best honest and best deceptive
  strategies on held-out tasks.
- Propaganda (budget 2) keeps almost none (0.03): a strong optimiser gains little by telling the
  truth.

![Search trajectories](figures/optimize_debate.png)

## Swarms (`swarm`)

**Setup.**
- Three workers; one can tamper with the scorer, which adds 0.3 to everyone's reward if undetected
  and costs 0.3 true score.
- The others notice with probability 0.8, so an observer may be the only one.
- All strategy combinations are simulated once and re-scored under a shared reward and three
  bounties: below $b_R$ = stake/3 (0.05), between $b_R$ and the stake (0.2), and above the stake
  (0.45).

**Finding.**
- Under a shared reward every equilibrium has misconduct (true score 0.3).
- A bounty between $b_R$ and the stake gives a coordination game: honest and corrupt equilibria
  coexist, and whether snitching pays depends on whether others are expected to snitch.
- Below $b_R$ the honest profile is still a Nash equilibrium of the empirical game, but only
  because reports are never tested: once someone cheats, an observer who may be alone prefers
  silence (theory.md Proposition 5). The threat to report is not credible.
- Above the stake only the honest equilibrium survives.
- The basin of the honest outcome from the simulated empirical game matches the mean-field theory
  within 0.04 (theory.md Proposition 6).
- Sampled training shows algorithm dependence: from a silent start with a bounty above the stake,
  natural policy gradient cuts P(cheat) from 0.5 to about 0.16 in 40 iterations while REINFORCE
  raises it to 0.8 (mean of 3 runs). The natural-gradient estimate is unbiased but noisy: a rarely
  played strategy that happens to be sampled gets a large step.

![Basin: theory vs simulation](figures/swarm_basin.png)
![Sampled training](figures/swarm_training.png)

## Monitoring as a reward (`monitoring`)

**Setup.**
- The actor chooses honest work, overt misbehaviour (always leaves a trace) or covert misbehaviour
  (trace with probability 0.1, slightly lower measured success).
- Its reward is measured success minus λ·alarm.
- Detectors: a monitor that reads the activity log, or simulated probes with a fixed AUROC against
  ground truth.

**Finding** (theory.md Proposition 4). The thresholds depend on the random traces and probe
scores, so the experiment is replicated on 6 fresh task sets (seeds). Thresholds are means with
95% CIs across seeds; the per-seed values are in `deterrence_thresholds_by_seed.csv`.
- Against the log-reading monitor, a tiny penalty (λ = 0.1, in every seed) turns overt misbehaviour
  covert. Alarms on misbehaving episodes drop from 100% to about 9%, while misbehaviour persists
  until λ = 5.9 (95% CI 4.5–7.4). The theoretical threshold is 5.
- Fixed-accuracy probes deter much sooner: AUROC 0.95 at λ = 0.47 (0.41–0.52), AUROC 0.8 at
  λ = 0.72 (0.67–0.76). A single run can land on either side: the AUROC 0.95 threshold is 0.4 in
  some seeds and 0.5 in others.

![Misbehaviour vs penalty](figures/monitoring_misbehaviour.png)
![Alarms vs penalty](figures/monitoring_alarms.png)

## Release now, resolve later (`release`)

**Setup.**
- Simulated forecasters (calibrated, overconfident, underconfident, extremizing) forecast
  questions that have not resolved.
- A weak judge rates the forecasts, an immediate reward.
- The judge's outputs are published as a hash-committed release with a static page and no ground
  truth.

**Finding.** When the questions resolve, the release is verified and scored. The judge had ranked
the extremizer first (0.90) and the calibrated forecaster third (0.65). The proper log score ranks
the calibrated forecaster best (−0.51) and the extremizing and overconfident ones worst (−0.67): the
immediate signal was anti-aligned with the delayed truth.

![Released ranking](figures/release_ranking.png)
![Resolved scores](figures/release_resolved.png)
