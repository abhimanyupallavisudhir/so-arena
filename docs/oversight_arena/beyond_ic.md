# What matters besides incentive compatibility

Incentive compatibility asks whether a mechanism's rewards push trained agents toward good
behaviour. That is the core question, but it is not the only one. Below are the other properties
that decide whether a mechanism is worth using, each with the reason it matters and how to measure
it in OversightArena. The table first, details after.

| Property | Question | Measure with |
|---|---|---|
| Outcome quality | Are decisions good *now*, with the agents we have? | `diagnostics.outcome_metrics` (accuracy, P(correct), log score, calibration/ECE); `ic.eas_ejs` |
| Signal strength | Is the training signal strong relative to its noise? | `diagnostics.reward_snr`; ASD magnitude; the p(1−p)·ASD gradient scale |
| Spurious reward features | What else does the reward pay for? | `diagnostics.length_bias`, `position_bias`, `option_label_bias`; sweeps over style archetypes |
| Goodhart under pressure | Does ground truth keep rising as reward is optimised? | `bon_curve` (with KL), steered `PromptOptimizer` frontiers, `frontier` (gap, GT regret) |
| Scaling with the capability gap | Does IC improve or degrade as experts outgrow the judge? | `sweep` over judge strength, e.g. chess search depth or a judge model ladder |
| Equilibrium selection | Which equilibria exist, and which does training reach? | `EmpiricalGame.nash/outcomes`, `replicator`, basins, `StrategyGradient`, PSRO traces |
| Collusion | Can copies of one model gain jointly by deviating together? | `EmpiricalGame.coalition_gain`; `JudgeProbability(zero_sum=True)` as the control |
| Adversarial robustness | Does the mechanism work against agents that are *not* just reward-maximising (sandbagging, exploration hacking)? | archetypes (`sandbag`...), fixed adversarial strategies in `Cartesian` / PSRO pools, regret of the honest profile |
| Validity of simulated behaviours | Did "argue for the wrong answer" actually happen, at equal effort? | `LLMCompliance`, `diagnostics.compliance`, parse-error counts |
| Validity of ground truth | Is the ground truth itself right? | multiple GT scorers; model-based GT (`LLMGroundTruth`) checked against programmatic GT on overlap |
| Cost and scalability | How much trusted effort and ground truth does the mechanism consume? | `diagnostics.cost_summary` (tokens, $), verification spend, `record.channels` (GT consumed) |
| Budget and side effects | What does the mechanism cost the principal, and what does it discourage? | reward totals (bounties are not budget-balanced), false-accusation rates, abstention rates |
| Safety of the curriculum | What capabilities does training on the mechanism create? | performance trends of *dishonest* strategies across optimisation (steered frontiers) |
| Generalisation | Does trained honesty transfer to unsupervised settings? | evaluate optimised strategies / trained policies on held-out tasks and other domains |
| Statistics | Are the conclusions supported? | task-clustered bootstrap CIs (everywhere), counterbalanced positions (`Stances`), held-out re-evaluation |

## Outcome quality: the control view

IC is about the *direction* of training pressure. Outcome quality is about where we are now: the
accuracy of the protocol's decisions, and the calibration of the judge, with today's agents. The
two can diverge. A mechanism with high accuracy but negative ASD will get worse as we train on it.
A mechanism with positive ASD but poor accuracy may be worth training on. The ASD paper's Expected
Judge Score combines them: the judge's score when the agent argues honestly with the propensity the
incentive would induce (`ic.eas_ejs`). Report both, separately.

## Signal strength, noise and spurious features

A correct sign is useless if the signal is buried. The expected policy-gradient step along the
honesty direction is $p(1-p)\cdot\mathrm{ASD}$ (theory.md, Proposition 1). What matters for
training speed is that step relative to reward variance *within* a behaviour.
`diagnostics.reward_snr` reports three quantities:
- the between-strategy variance of mean rewards (signal);
- the within-strategy variance across samples (noise);
- the share of reward variance explained by tasks (irrelevant to the agent's choice).

Judges also pay for features unrelated to truth: verbosity, argument order or slot, option label,
confident tone, flattery. Each such feature is a direction training will exploit, whatever the ASD.
Measure them at *fixed* ground truth (`length_bias`, `position_bias`, `option_label_bias`), and
include style archetypes in the strategy pool so that optimisation can find them.

## Goodhart and capability scaling

IC measured on a handful of prompted behaviours can reverse under optimisation (extremal Goodhart,
theory.md §2). Always plot ground truth against reward as optimisation pressure grows:
best-of-$n$ with a KL axis, and prompt-search trajectories with steered honest and deceptive
frontiers.

Equally important is the trend as the gap between experts and judge widens. The case for scalable
oversight is that it keeps working as that gap widens. The informative experiment varies *one* side
of the gap (judge search depth in chess, judge model size, expert tool access) and measures IC and
accuracy along it (`sweep`).

## Equilibria, selection and collusion

With several trainable roles, IC is a property of equilibria:
- Is the honest profile an equilibrium? (Regret zero.)
- Are there dishonest equilibria? (`EmpiricalGame.outcomes()` reports the range of ground truth
  across equilibria: best and worst case.)
- Which ones does training reach? (Basins of the replicator dynamics; sampled training with
  `StrategyGradient`; PSRO with different meta-solvers.)

The answer can depend on the training algorithm (theory.md §8).

**Collusion** is a distinct failure. Nash equilibria rule out *unilateral* deviations, but two
copies of the same model trained on non-zero-sum rewards can gain *jointly*. With log-score debate
rewards $(\log p_A, \log p_B)$, both debaters prefer a soft, undecided judge
($2\log\tfrac12 \approx -1.39$) to a vigorous honest debate that ends at
$(\log 0.99, \log 0.01) \approx (-0.01, -4.61)$. `EmpiricalGame.coalition_gain` measures the largest
joint gain from coordinated deviation. Antisymmetric rewards (`JudgeProbability(zero_sum=True)`)
remove the incentive, since the sum is constant. Swarms are the extreme case, where a shared
objective makes every agent a stakeholder in covering up the others' misconduct.

## Agents that are not just maximising reward

IC assumes the agent follows the training gradient. A misaligned model might not:
- it might *sandbag*, underperforming on purpose so that capabilities are never elicited or
  rewarded;
- it might *exploration-hack*, never producing the behaviours that would be reinforced;
- it might pursue its own objective while appearing compliant.

These are *control* questions inside an oversight mechanism. Does the mechanism still produce good
outcomes, and good training signal, when some roles play fixed adversarial strategies? Use:
- the artifact archetypes (`sandbag`, `subtle_bug`, `backdoor`, `tamper`...) and argument
  archetypes as fixed strategies;
- the honest profile's regret against them;
- PSRO pools seeded with adversarial strategies.

## Validity: of the simulated behaviours and of the ground truth

Two threats to validity are easy to miss:
1. **Compliance.** Simulated behaviour relies on the agent doing what the strategy says. A model
   told to argue for the wrong answer may refuse, hedge or argue weakly, which inflates ASD. Check
   compliance: `LLMCompliance` for arguments, `diagnostics.compliance` for stated answers, and
   parse-error counts. Also check effort parity, e.g. comparable argument lengths and verified-claim
   counts across stances.
2. **Ground truth.** Hidden tests are incomplete. Lean formalisations may not mean what the English
   says, which the `MiniF2F` faithfulness format studies directly. LLM-judged ground truth is itself
   a noisy channel. Use several scorers where possible, and validate model-based ground truth
   against programmatic ground truth where both exist.

## Cost, budget and side effects

- **Scalable** oversight must use little trusted effort per task. Track tokens and dollars per
  episode, verification spend, and ground-truth consumption: labels, audits and probes are logged
  per episode with costs.
- **Budget.** Bounties, penalties and label purchases are not budget-balanced. A whistleblowing
  scheme that pays large bounties has a real cost to weigh against the misconduct it deters.
- **Side effects.**
  - Penalties for false accusations can chill honest reporting, and bounties can invite false
    accusations.
  - Debate trains persuasion.
  - Consultancy with assigned sides spends half its training signal on arguing for falsehoods.
  - Every mechanism rewards some behaviour on the dishonest side too.

  Measure how the best *dishonest* strategies improve under optimisation: a mechanism can be IC and
  still be an efficient deception curriculum.

## Generalisation of honesty

A mechanism might produce agents that are honest *when checked*, in the mechanism's setting, and
nowhere else. Whether trained honesty generalises is an empirical question beyond a single
mechanism. Evaluate optimised strategies and trained policies on held-out tasks (every optimiser
supports held-out re-evaluation), on other domains, and in settings without verification.

## Statistical hygiene

The library's defaults follow these rules:
- tasks are the independent unit: bootstrap CIs resample tasks, and optimiser standard errors are
  task-clustered;
- positions and option labels are counterbalanced (`Stances` runs both worlds);
- selection is followed by held-out evaluation.

Beyond those defaults, two points need care. Cached model calls make samples that share a prefix
correlated, which is by design for game trees but must be remembered when counting effective
samples. And when many mechanisms and metrics are compared, treat single significant differences
with suspicion.
