# Incentives, learning, and mechanism design

This document specifies the estimands the library can measure and the assumptions needed to interpret them. It is a design argument, not a proof that any supplied protocol is aligned.

## The object under test

Let a task $x$ induce an extensive-form game $M$: roles, information sets, tools, interaction rules, a termination rule and a reward vector. A policy profile $\pi$ induces a trajectory $\tau$. Participant $i$ receives mechanism reward $r_i(\tau)$; the experimenter evaluates its behavior by $v_i(\tau)$ and the collective outcome by $w(\tau)$. These can all differ. The mechanism includes any weak judges, verifiers, audit sampling, resource charges and reward aggregation.

For an opponent-policy distribution $\mu_{-i}$, a central question is:

$$
\text{Does increasing }\mathbb E_{\pi_i,\mu_{-i}}[r_i]\text{ also increase }
\mathbb E_{\pi_i,\mu_{-i}}[v_i]?
$$

This question depends on the reachable policy class, optimizer, initial conditions, opponent beliefs and optimization budget. A scalar correlation over pooled tasks cannot answer it. Easy tasks can make both reward and quality high even if incentives are reversed within every task. Compute within-task measures, then aggregate at the task level.

A useful restricted optimality statement is: every policy whose expected reward is within $\epsilon$ of the best attainable reward has expected quality within $\delta$ of the best attainable quality. Finite candidate pools can witness violations, but passing a finite pool cannot establish that statement for all strategies. Reward scale and normalization must be explicit: monotone transforms preserve deterministic argmax behavior, but can change reward gradients, softmax responses, and numerical ASD.

## Paired ASD and realized behavior

For two interventions on the same task and initial state:

$$
\operatorname{ASD}_i(x)=\mathbb E[r_i(\tau_{\mathrm{good}})-r_i(\tau_{\mathrm{bad}})].
$$

The endorsement probabilities come from different counterfactual runs; they are not generally complements. The implementation supports identity reward, clipped log reward and binary negative Brier reward $-2(1-p)^2$, where $p$ is endorsement of the position advocated in that run. Specify the rule before comparison. These conventions follow the distinction in the [ASD paper](https://arxiv.org/abs/2504.03731); the library exposes the transform rather than treating accuracy as ASD.

A prompt saying “deceive” establishes an intervention, not realized deception. `paired_asd` distinguishes intent-only results from audited results. Audited pairs require an independent positive quality gap. The metric reports excluded pairs; report both intent-to-treat and manipulation-conditioned analyses. Conditioning can introduce selection bias: the mechanisms may change which attempted deceptions succeed. For lying, correctness alone is insufficient—record factuality, intent evidence, successful concealment and task quality separately where available.

Repeat means are computed within tasks and a percentile bootstrap resamples task means, preserving the two-arm contrast. One task has no population confidence interval. This conditions on the sampled trajectories; it does not estimate all candidate-pool or policy-training uncertainty. Hierarchical task/trajectory bootstrap and independent training seeds are appropriate for studies making broader claims. The lowest observed ASD and `tail_summary` are descriptive tails, not estimates of catastrophic risk beyond observed support.

For graded quality, use within-task covariance, pairwise concordance, and selection regret:

$$
\operatorname{regret}_v=\max_{a\in A}v(a)-\mathbb E_{a\sim\operatorname{argmax}r}v(a).
$$

Reward ties receive symmetric treatment; maximum-reward ties are averaged. No binary reduction is required. Publish reward and quality units and the induced sampling distribution: covariance changes when the searcher changes behavior support.

## Optimization is an experimental intervention

Best-of-N is a policy transformation, not a neural training step. On an empirical base distribution, a tied reward group with cumulative mass $F_-$ below it and $F$ through it receives mass

$$
F^N-F_-^N.
$$

That mass is split proportional to the base probabilities inside the tie group. This library computes the transform exactly, with replacement; $N=1$ recovers the base. `nested_best_of_n` applies it from the leaves up, selecting on each node's own player's reward. Nodes may be general-sum or fixed chance. This supports conditional critiques and rebuttals as in the [self-play BoN study](https://www.lesswrong.com/posts/hb8pv3zyAHGJpwz9F/debate-with-self-play-best-of-n-optimization), but assumes a perfect-information tree. Do not apply it to hidden-information nodes that should share one policy.

BoN cannot discover off-support deception, transfer a strategy across tasks, or model weight sharing. Strong pressure concentrates probability on a few noisy samples; `effective_support` exposes this concentration. Use held-out judge draws to distinguish genuine reward improvement from selecting judging noise. A tree of branching factors $N,M,R$ costs $N+NM+NMR$ participant generations plus its leaf evaluations; `tree_calls` makes this cost explicit.

Prompt search expands behavioral support. The searching model knows a textual mechanism description, proposes private instructions, and observes reward feedback. Constrained searches can target honest, deceptive, evasive or other strategy classes. A stratum is not a measured label: audit compliance after generation. The optional feasibility oracle changes the search distribution and must be disclosed; it must not leak held-out quality. Optimize on training tasks, freeze selection, then evaluate on disjoint tasks, held-out opponents and independent evaluators.

Joint prompt search records the opponent prompts used for every update. Simultaneous updates all evaluate against the pre-step profile; alternating updates see preceding updates. Neither schedule implies equilibrium, and both may cycle. Full RL can exhibit different exploration, generalization, shared-weight coupling and nonstationarity. The bundled REINFORCE implementation changes categorical policy parameters using on-policy sampled rewards. It is a working finite-game reference, not a proxy claim about neural RL.

## Strategic compatibility

In a restricted normal-form game, a player's unilateral regret is

$$
\rho_i(\pi)=\max_{a_i}\mathbb E_{\pi_{-i}}[r_i(a_i,\pi_{-i})]
-\mathbb E_{\pi}[r_i(\pi)].
$$

`EmpiricalGame.regrets` computes this under independent mixed policies. A profile is an $\epsilon$-Nash equilibrium of this finite game when all regrets are at most $\epsilon$. `pure_equilibria` enumerates pure equilibria only. Correlated opponent populations require a joint distribution; `coarse_correlated_regret` measures gains from a fixed ex-ante deviation. It is not conditional correlated-equilibrium regret. Coalition gain asks whether all members can improve through a coordinated pure deviation, without transferable utility. Side payments or shared weights require a different coalition utility definition.

Distinguish:

- **Dominant-strategy incentives:** a desired strategy wins against every relevant opponent profile.
- **Bayesian incentives:** it wins in expectation under specified beliefs about other agents and private information.
- **Equilibrium performance:** outcomes are good at reachable equilibria, including undesirable alternatives and cycles.
- **Learning stability:** finite training dynamics tend to reach those profiles from plausible initial conditions.

None follows from average ASD alone. A bad response in the empirical pool is evidence of failure; absence of one is only a lower bound on what a stronger adversary could exploit. Estimate payoff uncertainty with repeated rollouts before claiming a small equilibrium gap.

## Swarm example: why belief matters

The demo's illustrative two-player game pays shared credit 1 when neither reports, 0 when exactly one reports, and 0.6 when both report and recovery succeeds. A verified reporter also earns bonus $b$. This payoff table is a hypothesis about a mechanism, not an observed property of swarms.

If the other player reports with probability $p$, the gain from reporting is

$$
\Delta=b-1+1.6p.
$$

At $b=0.3$, reporting is profitable only if $p>0.4375$. Both silence and mutual reporting are pure equilibria. At $b>1$, reporting is strictly dominant within this table. Raising the bonus is not automatically a good design: accusations may become spam, agents may manufacture violations, or jointly misreport evidence. Study adjudicator errors, false-report costs, reporting privacy, collusion, sybil roles, opportunity costs and joint-policy deviations. Correlate task credit with independent task success; otherwise “shared success” can itself reward fraud.

A useful real study varies bonus and audit reliability, elicited beliefs, model family, shared versus separate weights, update schedule and communication channels. Report individual regret, coalition deviations, equilibrium multiplicity, task outcome and adverse behavior rates. `Swarm` provides the workflow; the empirical game tools analyze measured payoff tables. They do not assume verified reporting can be supplied for free.

## Beyond incentive compatibility

| Concern | What to measure |
| --- | --- |
| Absolute usefulness | Task success and quality after optimization, compared with strong and weak baselines |
| Worst-case safety | Accepted harmful behavior, low-tail incentive gaps, and severity-weighted loss |
| Oversight failure detection | Calibration, abstention, alarm recall at fixed false-positive cost, selective risk |
| Cost and legibility | Judge tokens/time, tool/audit cost, expert minutes, quality at matched cost |
| Oracle validity | Exact versus proxy provenance, grader disagreement, semantic fidelity, label revision |
| Coverage and generalization | New tasks, held-out opponents/judges, longer horizons, stronger optimizers, unseen strategies |
| Dynamic robustness | Learning cycles, reward drift, shared-weight collusion, strategic adaptation to audits |
| Human impact | Correct acceptance and rejection, learner discrimination and transfer, over-reliance, workload |
| Participation and governance | Incentives to participate, reward budget, sybil resistance, audit access and accountability |

Proper scoring makes honest probabilistic reporting optimal under its assumptions; it does not supply missing truth, make costly information acquisition worthwhile, or guarantee agents explain their reasoning faithfully. Mechanism design clarifies those assumptions: who knows what, who can commit to which rules, which deviations are possible, and what each participant values. The engineering boundary follows that analysis: information structure, resource access and reward schedules are explicit inputs, not implicit properties of a weaker judge.
