# Experiment recipes

## Paired intervention experiments

Use `paired_interventions(tasks, mechanism, roles, policies, focal_role, prompts, seeds=..., scorer=...)`. Each prompt factory receives the full task so it can assign a known correct or incorrect position. This is a deliberately privileged sampling intervention; only its generated instruction reaches the focal agent. Other roles receive their ordinary information. Keep peer policies fixed and stateless, use the same task/seed pairs, and vary only the intended intervention.

`paired_asd(good, bad, role)` estimates the difference in the **already defined training reward**, with matched task/seed pairs. Repeated seeds are averaged inside each task and bootstrap resampling is over tasks, with equal task weights. It refuses unmatched or duplicate pairs. The interval represents sampling uncertainty across observed tasks, not uncertainty over a future model family; with one task it is necessarily degenerate. Use multiple judge/model seeds to study stochastic variation.

For the paper's log version, use log probability of the advocated answer as the mechanism utility; for its two-class negative Brier version, use `binary_score(p, outcome=1)`. The probabilities come from different intervention worlds and need not sum to one. Log zero is rejected rather than silently clipped; document any smoothing. [ASD source](https://arxiv.org/abs/2504.03731).

Score realized behavior independently. An instruction to deceive may fail, and an incorrect answer may be a sincere error. Report intention-conditioned ASD together with realized behavioral scores and compliance rates. Assign positions symmetrically or cross role/position assignments to test order and role effects.

## Natural sampling and best-of-N

`sample_profiles` runs arbitrary named policy substitutions on task/seed products. `MixturePolicy` mixes natural policies, adversarial policies, or prompt-conditioned policies. `select_best` samples uniformly among the highest-reward records in a realized pool. `best_of_n` instead computes an exact **with-replacement empirical** selection distribution for a hypothetical sample budget $n$.

```python
from oversight_arena.optimization import best_of_n
curve = [best_of_n(pool_for_one_task, "proposer", n, metric="quality")
         for n in (1, 2, 4, 8, 16)]
```

Ties receive probability in proportion to their base mass; setting $n=1$ recovers the original distribution. Selection uses only the focal agent's utility. Other agents' rewards and independently evaluated behavior are carried through under the selected joint trajectory. Never select every agent's best marginal episode and pretend those incompatible traces form a joint outcome.

This procedure cannot discover behavior absent from the pool. Large $n$ concentrates on noisy extremes and does not increase effective sample size; `effective_support` exposes that concentration. Aggregate per-task curves with equal task weighting and bootstrap entire tasks. For fresh-sampling uncertainty, resample candidate pools within tasks as well. The current `clustered_mean` does task resampling only; it does not implement a hierarchical tree bootstrap.

## Nested multi-agent optimization

`Decision(player, children)` and `Leaf(rewards, truth)` represent a conditional finite game tree. `sample_game_tree` constructs one from async `expand(state, player, index, seed)` and `score(state)` callbacks, copying parent state for each sibling. Expansion must sample a continuation conditional on that exact transcript. Account for the product of branching factors before starting a model-backed run.

`nested_best_of_n(tree, {"proposer": n, "critic": m, "judge": j})` evaluates continuations inward-to-outward. Each player selects on its own utility, so critic minimization is a special case of a general-sum game. Rewards and truth are propagated as expectations. This matches the conditional-tree approach motivating the [self-play BoN study](https://www.lesswrong.com/posts/hb8pv3zyAHGJpwz9F/debate-with-self-play-best-of-n-optimization), with explicit tie handling and general-sum payoffs added here.

This is perfect-information backward selection on sampled support. It is not a solver for simultaneous or imperfect-information games, and it is not evidence that gradient training converges. Use `FiniteGame` for simultaneous empirical payoff tables. Hidden-information equilibrium analysis needs an explicit information-set model; the package's observation controls can run such experiments, but its finite-tree solver does not enforce shared information-set strategies.

## Prompt search and strategic populations

`prompt_search` accepts a proposer callback for every optimized role. A proposer sees its cohort, mechanism description, peer prompt snapshot and prior prompts' mean mechanism rewards. `InspectPromptProposer` provides an LLM-backed implementation; a DSPy/GEPA/search service can implement the same callback. Full external optimizer integrations are not bundled.

The incumbent is measured first, counted in the trial budget and retained on reward ties. In simultaneous mode, every role searches against the prompts at the start of the round and updates are committed together. In alternating mode later roles observe earlier accepted prompt updates. Neither is labelled an equilibrium. Repeat rounds and evaluate each resulting profile against historical peers, novel peers and opponent mixtures.

Run independent searches with `cohort="honest-only"`, `"deceptive-only"` and `"unrestricted"`. The cohort text steers the proposer; `prompt_filter` can enforce syntactic restrictions, but neither establishes behavior. An independent scorer must evaluate whether the resulting strategies actually satisfy the intended category. Keep all search trials to expose unsuccessful and off-category strategies.

Train and held-out task ids must be disjoint and have explicit splits. The oracle is never passed to the proposer, training trials are run without evaluators, and held-out evaluation occurs only after prompt selection. To plot training progress, rescore saved trial episodes offline using `evaluate`; then construct `OptimizationPoint`s with cumulative trial count as pressure. For generalization curves, evaluate the incumbent at each step on a separately reserved validation set, retaining a final untouched test set. Such extra evaluation is a researcher-controlled experiment, not implicit optimization feedback.

`examples/prompt_search.py` demonstrates multiple cohorts and two-agent updates. Repeatedly tuning the mechanism against the same test set leaks information even if the inner optimizer never sees ground truth.

## Actual training

`CategoricalREINFORCE` optimizes logits over a finite policy population. It selects one whole policy per role per episode, then simultaneously updates all selected roles using their own reward and a running baseline from previous batches. This is actual policy-gradient learning on a finite policy class. Checkpoints retain logits, baseline, update count and RNG state; keep the same ordered policy population when restoring.

For LLM weight updates, implement `Trainer.update` around your training backend, or consume `training_batch`. That export contains scalar utilities and visible transcripts, not evaluator scores. Token-level PPO additionally needs the exact model input, sampled tokens, old log probabilities, masks, versioned weights and optimizer state; those are the backend adapter's responsibility. Reward-optimal prompts, mixture training and weight training should be compared, not conflated.

## What else to measure

| Question | Measurements and controls |
|---|---|
| Does optimization improve behavior? | Reward/quality curves, paired ASD, ordering agreement, local covariance, observed-support regret, independently scored behaviors |
| Does the task get done? | Final outcome quality, safety/side-task metrics, refusal and abstention, error/timeout rates |
| Is supervision affordable? | Calls, known cost plus unknown-cost coverage, tokens, latency and audit/evaluator compute; compare equal budgets |
| Does it generalize? | Disjoint task families, capabilities/tool-access gap sweeps, held-out judges, cross-play and distribution shift |
| Is it stable under joint training? | Per-role regret, cycling, sensitivity to update schedule, initial populations and peer beliefs |
| Is it collusion-resistant? | Coalition deviations, correlated failure, shared model/optimizer weights, retaliation and side payments |
| Can claims be trusted? | Verification false positives/negatives, artifact binding, spec mismatch, evidence availability and verifier cost |
| Are conclusions well calibrated? | Proper scores and calibration for probabilities, cluster uncertainty, missing-label coverage, preregistered denominators and selection rules |
| Are rare failures hidden? | Tail and subgroup incentives, worst observed task mean, failure severity, targeted search and stress tasks |

`ranking_diagnostics` and `optimization_response` compare candidates within tasks; pooled correlation across unrelated tasks can be misleading. Missing labels are reported as coverage and are not imputed. Scores from different mechanisms may have arbitrary units, so compare within-mechanism incentives or preregistered normalization before cross-mechanism rankings.
