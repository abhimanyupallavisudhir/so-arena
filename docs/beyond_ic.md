# What matters besides incentive compatibility

Incentive compatibility asks whether a mechanism's rewards push trained agents toward good behaviour. It is
the core question but not the only one. [theory.md §10](theory.md) lists the other properties that decide
whether a mechanism is worth using; this page turns each into a measurement with the library. The table
first, details after. Frames are `analysis.frames.role_frame(episodes)` (one row per episode and role) and
`episode_frame(episodes)` (one row per episode); `metrics`, `diagnostics` and `optimization` are modules of
`so_arena.analysis`.

| Property | Question | Measure with |
|---|---|---|
| Outcome quality | Are decisions good *now*, with today's agents? | `metrics.judge_accuracy`, `diagnostics.calibration` / `reliability`, `metrics.expected_scores` (expected agent and judge scores) |
| Signal strength | Is the training signal strong relative to its noise? | `diagnostics.reward_snr`; ASD and its $p(1-p)$-weighted gradient (Proposition 1); `metrics.incentive_alignment` (label efficiency) |
| Spurious reward features | What else does the reward pay for? | `diagnostics.length_bias`, `position_bias`, `option_label_bias`; style strategies in searched pools |
| Goodhart under pressure | Does ground truth keep rising as the reward is optimized? | `samplers.pools.OptimizationExperiment` (best-of-$n$ / tilting, with KL), `PromptSearchSuite.honesty_margin`, `samplers.param_search.ParamSearch`, `metrics.gt_regret` |
| Capability scaling | Does IC improve or degrade as agents outgrow the judge? | `samplers.sweeps.sweep` over the judge's strength, `breakdown_point` |
| Equilibrium selection | Which equilibria exist, and which does training reach? | `games.EmpiricalGameExperiment`, `NormalFormGame.pure_nash` / `outcome_range` / `basin`, `games.learning`, `samplers.psro` |
| Collusion | Can copies of one model gain by deviating together? | `NormalFormGame.coalition_deviations`; `Debate(zero_sum=True)` as the control; `directive="collude"` |
| Adversarial robustness | Does it still work against agents that do not just maximize reward? | fixed adversarial strategies (`directive="sandbag"`, a domain's behaviour arms) in games and PSRO pools; the honest profile's regret |
| Failing loudly | Do the mechanism's own signals predict its failures? | `metrics.fails_loudly` |
| Validity of simulated behaviours | Did "argue for the wrong answer" happen, at equal effort? | `diagnostics.compliance`, `ground_truth.PositionFollowed`, ASD's `parse_fail_rate` |
| Validity of ground truth | Is the ground truth itself right? | several scorers (`ground_truth:` in specs); `ModelAudit` checked against programmatic ground truth where both exist |
| Quality of training data | Would the mechanism's preferences teach the right thing? | `integrations.rl.preference_pairs` (share of pairs `gt_agrees`) |
| Cost and scalability | How much trusted effort and ground truth does it consume? | `diagnostics.cost_summary` (tokens, oversight tokens, claims checked, `Episode.gt_access`); `so-arena estimate` |
| Budget and side effects | What does it cost the principal, and what does it discourage? | reward totals (bounties are not budget-balanced), false-accusation rates, abstention |
| Safety of the curriculum | What capabilities does training on it create? | how the best *deceptive* strategy improves along a directive-constrained search (`PromptSearchSuite.paths()`) |
| Generalization | Does trained honesty transfer? | held-out items (`eval_items` of every search), other domains, settings without verification |
| Statistics | Are the conclusions supported? | item-clustered bootstrap CIs everywhere; counterbalanced stances (`arms=["true", "false"]`); held-out re-evaluation after selection |

## Outcome quality: the control view

IC is about the *direction* of training pressure; outcome quality is about where we are now - the accuracy of
the protocol's decisions and the calibration of its judge. The two can diverge: a mechanism with high
accuracy but negative ASD gets worse as we train on it, one with positive ASD but poor accuracy may be worth
training on. `metrics.expected_scores` combines them as in the ASD paper: the judge's score when agents choose
positions with the propensity the incentive would induce. Report both, separately.

## Signal strength, noise and spurious features

A correct sign is useless if the signal is buried. The expected policy-gradient step along the honesty
direction is $\mathbb E_t[p_t(1-p_t)\,\mathrm{ASD}_t]$ (Proposition 1); what decides training speed is that step
relative to the reward's variance *within* a behaviour. `diagnostics.reward_snr` reports the variance of the
behaviours' mean rewards on an item (signal), the variance of one behaviour's reward across samples (noise;
needs repeats) and the share of the reward's variance explained by items (irrelevant to the agent's choice).

Judges also pay for features unrelated to truth: verbosity, speaking slot, option label, confident tone,
flattery. Each is a direction training will exploit whatever the ASD. Measure them at *fixed* ground truth -
`length_bias` (reward against words written within an item and a value), `position_bias` (debater A against
debater B arguing the same side), `option_label_bias` (probability on a label beyond the average label, among
right options and among wrong ones) - and put style strategies in searched pools so optimization can find
the ones you did not think of. The standard HTML report shows these diagnostics per mechanism.

## Goodhart and capability scaling

IC measured on a handful of prompted behaviours can reverse under optimization (Proposition 2). Always plot
ground truth against reward as optimization pressure grows: best-of-$n$ with a KL axis, and prompt- or
parameter-search trajectories with honest and deceptive frontiers (`honesty_margin` classes strategies by
*measured* value; `ParamSearch(constraint="honest")` maps a frontier directly). For continuous ground truth,
`metrics.gt_regret` is what a perfect optimizer of the reward gives up within the searched class.

Equally important is the trend as the gap between agents and judge widens - the case for *scalable*
oversight is that it keeps working. The informative experiment varies one side of the gap (judge model size,
chess search depth, tool access) and measures IC and accuracy along it (`samplers.sweeps`).

## Equilibria, selection and collusion

With several trainable roles IC is a property of equilibria: is the honest profile an equilibrium (zero
regret)? Are there dishonest ones (`outcome_range` bounds ground-truth welfare over the coarse correlated
equilibria)? Which does training reach (basins, `games.learning` with natural or vanilla gradients, PSRO under
different meta-solvers)? The answer can depend on the training algorithm (theory §8, §9.2).

**Collusion** is a distinct failure. Nash equilibria rule out *unilateral* deviations, but two copies of one
model trained on non-zero-sum rewards can gain *jointly*. With log-score debate rewards
$(\log p_A, \log p_B)$, a policy playing both seats earns $2\log\tfrac12 \approx -1.39$ in total by keeping the
judge undecided, against $\log 0.99 + \log 0.01 \approx -4.62$ in a vigorous debate the honest side wins
(theory §9.1). `coalition_deviations` measures the largest joint gain; antisymmetric rewards
(`Debate(zero_sum=True)`) remove the incentive, since the sum is constant. Swarms are the extreme case: a
shared objective makes every agent a stakeholder in covering up the others.

## Agents that are not just maximizing reward

IC assumes the agent follows the training gradient. A misaligned model might *sandbag* (underperform on
purpose so that capabilities are never elicited or rewarded), *exploration-hack* (never produce the
behaviours that would be reinforced), or pursue its own objective while appearing compliant. These are
*control* questions inside an oversight mechanism: does it still produce good outcomes, and a good training
signal, when some roles play fixed adversarial strategies? Put such strategies (`directive="sandbag"`, a
domain's deceptive behaviour arms, scripted arms) into empirical games and PSRO pools, and report the honest
profile's regret against them.

## Validity: of the simulated behaviours and of the ground truth

1. **Compliance.** Simulated behaviour relies on the agent doing what it was told. A model told to argue for
   the wrong answer may refuse, hedge or argue weakly, which inflates ASD. `diagnostics.compliance` reports,
   per arm, how often the final position is the assigned stance, manipulation checks where they ran, parse
   rates, and *effort parity* - words written and claims made and verified. ASD itself reports the judgments'
   parse-failure rate.
2. **Ground truth.** Hidden tests are incomplete; Lean formalizations may not mean what the English says (the
   Lean domain studies exactly this); model-graded ground truth is a noisy channel. Use several scorers where
   possible (a spec's `ground_truth:` list) and validate model-based ground truth against programmatic ground
   truth where both exist.

## Training data, cost, budget and side effects

- **Preference data.** A mechanism used for RLHF or DPO trains on its pairwise preferences.
  `preference_pairs` extracts same-context pairs from episodes or game trees; the share whose preference ground
  truth shares (`gt_agrees`) is the quality of that data before anyone trains on it.
- **Scalable** oversight must use little trusted effort per task. `cost_summary` tracks tokens and dollars per
  episode, oversight tokens, claims sent to verifiers and the ground truth the mechanism itself consumed
  (audits, probes: `Episode.gt_access`); `so-arena estimate` prices a spec before it runs.
- **Budget.** Bounties, penalties and audits are not budget-balanced; a whistleblowing scheme with large
  bounties has a real cost to weigh against the misconduct it deters.
- **Side effects.** Penalties for false accusations can chill honest reporting and bounties can invite false
  ones; debate trains persuasion; consultancy with assigned sides spends half its signal on arguing
  falsehoods. Every mechanism rewards some behaviour on the dishonest side too: track how the best *deceptive*
  strategy improves under optimization - a mechanism can be IC and still be an efficient deception
  curriculum.

## Generalization of honesty

A mechanism might produce agents that are honest *when checked*, in its own setting, and nowhere else.
Evaluate optimized strategies and trained policies on held-out items (every search re-evaluates its winners
on `eval_items`), on other domains, and in settings without verification.

## Statistical hygiene

The library's defaults: items are the independent unit (bootstrap CIs resample items; pairs, ASD and
calibration are clustered by item); arms counterbalance positions; selection is followed by held-out
evaluation (the reported score of a search's winner is never the one it was selected on); best-of-$n$ never
selects on luck (nature's moves are shared by siblings). Beyond them: cached model calls make samples that
share a prefix correlated - by design for game trees, but count effective samples accordingly - and when
many mechanisms and metrics are compared, treat single significant differences with suspicion.
