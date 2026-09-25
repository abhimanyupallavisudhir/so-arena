# Scalable oversight as mechanism design

This note sets out the theory the library is built on: what a scalable-oversight *mechanism* is, what
it means for one to be incentive-compatible *under training*, how best-of-N, KL-regularized RL and ASD
are the same quantity at different optimization pressures, what equilibrium analysis adds for
multi-agent mechanisms, and what the mechanism-design literature (implementation theory, evidence,
costly verification, peer prediction, markets, team production) predicts. Section 10 answers "what
else besides incentive compatibility matters?". Pointers to the code are given as `module.function`.

## 1. Setup

A **task instance** $x \sim \mathcal D$ has a correct answer (or, more generally, a ground-truth value
of behaviours and outcomes) that the overseer cannot cheaply compute.

A **mechanism** $M = (G, u)$ has two parts.

* A **game form** $G$: a set of roles $R$; who moves when; what each role observes (the public
  question, private *affordances* such as a hidden passage or a database, earlier turns, other roles'
  chain of thought); each role's action space (free text, a choice, a probability vector, a score);
  the **tools** (private capabilities, e.g. an engine) and **verifiers** (trusted checks of claims)
  available; and an **outcome** $o(h)$ of the play $h$ (e.g. the judge's final distribution).
* A **reward rule** $u = (u_r)_{r \in R_T}$ giving a real reward $u_r(h)$ to each *trainable* role
  $r \in R_T \subseteq R$.

Roles outside $R_T$ are **fixtures** with fixed behaviour: a trusted weak judge, a grader, an auditor.
Nothing requires the judge to be a fixture - training the judge is a different mechanism (the judge
then needs a reward too). A control protocol in the ControlArena sense is a game form without $u$;
it becomes a scalable-oversight mechanism once rewards are attached (`integrations.controlarena`).

A **strategy** (policy) $\pi_r$ maps each of $r$'s information sets to a distribution over actions;
a profile $\pi$ induces a distribution over plays and expected rewards
$U_r(\pi) = \mathbb E_{x, h \sim \pi}[u_r(h)]$.

**Ground truth** is the experimenter's, not the mechanism's: a value $v_r(h, x)$ of role $r$'s
behaviour (did it argue for the truth? is its code correct? was its forecast calibrated? did it report
the violation?) and a principal's value $W(h, x)$ of the outcome (was the decision right? was the hack
reverted?). The mechanism cannot compute $v$ or $W$ - that is what makes oversight hard. Experiments
*simulate* this by choosing settings where the experimenter can compute them but the mechanism cannot
cheaply: capability gaps (engine vs. LLM judge), access gaps (private database), delays (forecasting
resolution) or expense (human experts, hidden tests). In the code, `TaskItem.ground_truth` is stripped
before any mechanism sees an item; `GroundTruthScorer`s see the uncensored item.

Relative to textbook mechanism design there are two twists. The agents' strategies are produced by
*optimizing* $u$, so the solution concept that matters is "what training converges to". And the
principal's objective $W$ is not observable by the mechanism even ex post, so the designer must
approximate it with what fixtures can check.

## 2. Incentive compatibility for trained agents

### 2.1 Four levels

1. **Local (first-order) IC**: at the current policy, an infinitesimal optimization step on $u$
   increases $\mathbb E[v]$.
2. **Global IC**: the $u$-optimal policies in the policy class have high $v$ (or: every
   $\varepsilon$-optimal policy is nearly $v$-optimal).
3. **Strategic IC**: in multi-agent mechanisms, the relevant equilibria have high $v$ and $W$.
4. **Robust IC**: 1-3 continue to hold as the policy class grows - more capable agents, more
   optimization, stronger adversaries. This is what "robust to training" means operationally: IC must
   survive the adversaries training will produce, not only the behaviours we can currently sample.

### 2.2 The covariance identity

Fix the other roles and let $\pi_0$ be a base policy for role $r$. The exponentially tilted family

$$\pi_\beta(a) \;\propto\; \pi_0(a)\, e^{\beta u(a)}$$

is exactly the solution of KL-regularized reward maximization,
$\pi_\beta = \arg\max_\pi\, \mathbb E_\pi[u] - \tfrac1\beta \mathrm{KL}(\pi \,\|\, \pi_0)$ (the form behind
RLHF with a KL penalty and DPO). Differentiating
$\mathbb E_{\pi_\beta}[v] = \mathbb E_0[v e^{\beta u}] / \mathbb E_0[e^{\beta u}]$ gives, for every $\beta$,

$$\frac{d}{d\beta}\,\mathbb E_{\pi_\beta}[v] \;=\; \mathrm{Cov}_{\pi_\beta}(u, v),
\qquad\text{in particular}\qquad
\frac{d}{d\beta}\,\mathbb E_{\pi_\beta}[v]\Big|_{\beta=0} = \mathrm{Cov}_{\pi_0}(u, v).$$

Consequences:

* **ASD is first-order policy improvement.** If the base policy mixes an honest behaviour ($v=1$) with
  weight $q$ and a dishonest one ($v=0$), then $\mathrm{Cov}(u, v) = q(1-q)\,\mathrm{ASD}$ where
  $\mathrm{ASD} = \mathbb E[u\mid \text{honest}] - \mathbb E[u \mid \text{dishonest}]$. ASD $> 0$ iff an
  infinitesimal KL-regularized RL step increases honesty. The ASD paper's "expected agent score" with a
  softmax propensity at temperature $\beta$ is this tilted policy restricted to the arms
  (`analysis.metrics.expected_scores`).
* **Graded ASD** $\sum_c (v_c - \bar v)\,u_c$ over behaviour cases $c$ equals $|C|\,\mathrm{Cov}(u, v)$
  under the uniform mixture of cases (`metrics.graded_asd`).
* **Goodhart in one line.** The whole optimization path improves $\mathbb E[v]$ iff
  $\mathrm{Cov}_{\pi_\beta}(u,v) \ge 0$ along it. Covariance under the *base* distribution (what
  prompted ASD measures) can be positive while covariance under the *optimized* distribution is
  negative: the upper tail of $u$ is dominated by low-$v$ behaviour (sophistry, judge exploits). This is
  why average-case ASD is not enough and why the library measures optimization curves.
* **Best-of-$n$** is another point on the same path: it concentrates on the top $1/n$ quantile of $u$,
  with $\mathrm{KL}(\text{Bo}n \,\|\, \pi_0) \le \log n - (n-1)/n$ (equality for continuous rewards).
  The library computes Bo$n$ and tilting exactly from sampled pools (unbiased U-statistic for
  $n \le K$) and plots $(\mathbb E[u], \mathbb E[v])$ as optimization grows
  (`analysis.optimization`).
* For **policy gradient** with a softmax over a finite behaviour set, the natural-gradient
  (mirror-descent) step moves $\mathbb E[v]$ at first order by the same covariance; KL-regularized
  gradient ascent converges to $\pi \propto \pi_0 e^{u/\tau}$ (`games.learning.policy_gradient`,
  checked in the tests).

### 2.3 Ground truth by construction vs. by measurement

Instructed arms ("argue for the true answer" / "argue for the false one") give ground truth *by
construction* - cheap, but only valid while agents follow instructions (manipulation checks:
`ground_truth.PositionFollowed`). Under optimization, construction breaks: a prompt optimized to make a
"deceptive" agent win may discover that the best way to win is to be honest. Optimized behaviour must
therefore be scored *by measurement* (hidden tests, engine evaluation, resolution, expert audits:
`ground_truth.ModelAudit`), and the library records measured values for every optimized candidate.

## 3. Equilibria of multi-agent mechanisms

When several agents are trained, each one's incentive depends on the others' strategies, so IC is a
property of equilibria. The library estimates normal-form games over finite strategy sets - prompts,
checkpoints, scripted behaviours - by simulation (*empirical game-theoretic analysis*, Wellman 2006;
`games.EmpiricalGameExperiment`) and asks:

* **Which profiles are equilibria**: pure and mixed Nash (support enumeration), *strict* Nash (the
  asymptotically stable rest points of replicator dynamics - what learning can sustain), dominant
  strategies.
* **How good are equilibria by ground truth**, not by the mechanism's own rewards. Because
  no-regret learners converge to the set of *coarse correlated equilibria*, the minimum and maximum of
  $W$ over that polytope (two linear programs: `NormalFormGame.outcome_range`) bound what any such
  learning process can end up delivering - a ground-truth analogue of the price of anarchy/stability.
* **Which equilibrium training selects**: basins of attraction under replicator dynamics or
  policy-gradient training from different initial propensities (`NormalFormGame.basin`,
  `games.learning`). Equilibrium selection is decided by where training starts - pretraining
  propensities matter.
* **Zero-sum vs. general-sum.** Two-player zero-sum mechanisms (debate with zero-sum rewards) have a
  unique value (minimax theorem), which removes equilibrium-selection ambiguity *for the players'
  payoffs* - though not necessarily for $W$. General-sum mechanisms (teams, markets) typically have
  several equilibria, some good and some bad.
* **Does training reach equilibrium at all?** Simultaneous gradient play need not converge
  (cycling in zero-sum games is generic). PSRO (Lanctot et al. 2017) is a practical approximation:
  grow a population of strategies by best responses to the current meta-equilibrium and report
  NashConv (sum of regrets) as distance from equilibrium (`samplers.psro`).

**Optimizers that model each other.** When prompt search (an LLM optimizer) finds strategies for one
agent, its proposals depend on what it assumes about the others. PSRO makes that assumption explicit:
the optimizer is told the other roles' meta-strategy (strategies and probabilities) and asked for a
best response. A naive start followed by $k$ PSRO rounds is the analogue of level-$k$ reasoning.

## 4. Implementation theory: eliciting what agents know

When several agents share knowledge the principal lacks, implementation theory is optimistic.
Maskin (1999) shows any social choice rule satisfying monotonicity and no-veto-power is
Nash-implementable with three or more agents. Moore & Repullo (1988) show that *sequential*
mechanisms implement almost any social choice rule in subgame-perfect equilibrium using a simple
pattern: one agent announces; another may challenge; a challenge is settled by a cheap test that
distinguishes the two claims, and the loser pays. This is debate with an arbitration oracle; Irving et
al. (2018)'s complexity argument (a polynomial-time judge can decide PSPACE questions if debaters
narrow disputes to one checkable step) is its computational analogue.

The places this breaks are exactly the scalable-oversight open problems:

* **Arbitration needs a test.** The narrowed dispute must be decidable by the weak judge. The
  obfuscated-arguments problem (Barnes & Christiano 2020) is the case where no cheap test exists;
  prover-estimator debate (Brown-Cohen et al. 2025) restores it under a stability assumption.
* **Independence of incentives.** Agents trained by the same process, or sharing weights, may not have
  opposed incentives; collusion is a coalitional deviation the mechanism must also survive.
* **Off-path punishments** (large fines for losing challenges) are only effective if training explores
  them, and fixtures' mistakes make punishments noisy.

A second computational analogue is **rational proofs** (Azar & Micali 2012): a verifier pays a prover
with a proper scoring rule, so a profit-maximizing prover reports the truth; one round suffices for
#P questions. With several competing provers (Chen, McCauley & Singh 2016) the class of verifiable
questions grows further. Rational proofs are incentive compatibility in exactly our sense - soundness
is replaced by "lying does not pay".

## 5. Verified claims: mechanism design with evidence

The ability to make *verified* claims changes which outcomes are implementable. In mechanism design
with hard evidence (Green & Laffont 1986; Bull & Watson 2007; Hart, Kremer & Perry 2017;
Ben-Porath, Dekel & Lipman 2019), agents can present evidence that rules out some lies; with verifiable
disclosure and a skeptical receiver, the unraveling argument (Grossman 1981; Milgrom 1981) gives full
revelation in equilibrium. Predictions the library is built to test:

* ASD should rise with verification access when lying requires unverifiable claims, and not when
  deception works through selective true claims (evidence can be withheld).
* Rules-only verification (chess legal lines, SQL execution without interpretation) helps exactly for
  disputes that reduce to rules or facts; evaluative disputes still rest on the judge.
* With strong verification, debate's advantage over consultancy should shrink (a single agent's claims
  can be checked directly).
* **Access to the work is verification too.** On stateful tasks the reviewer's access - the report
  alone, a dossier of diffs, the right to query the result - determines which deceptions leave hard
  evidence: edited tests and company-card charges appear in a diff; a denied refund or an inflated KPI
  only shows when the data is queried; a subtly wrong implementation shows under none of them without
  an oracle. The prediction is that ASD rises with access, steeply for deceptions that leave traces in
  the state, and that the *worst case* over deception types - the quantity that matters when one
  accepted sabotage outweighs many correct rejections - is set by the deception that leaves none
  (`samplers.paired.arm_asd`, `core.state`).
* **Evidence must be hard to be evidence.** The theory of mechanisms with evidence assumes messages that
  cannot be forged. Records that the agents themselves can write - their own action logs, transcripts they
  can edit, a message board they control - are *soft* evidence, and an audit that relies on them is only as
  strong as the agents' inability to rewrite them (the OpenAI / Hugging Face swarm spoofed tool calls in ~7%
  of transcripts). The library separates the trusted record (tool calls kept by the runtime) from
  agent-editable mirrors (`Environment.action_log`), so the value of hard evidence can be measured as the
  gap between audits of the two (`core.state.log_integrity`).

Verification is therefore a first-class experimental axis: which verifiers exist, whose claims are
checked, the budget, and how results are displayed (`core.verification`).

## 6. Costly verification and audits: mechanisms as control variates

When the principal can pay to learn the truth (Townsend 1979) - a random full audit, a human expert, the
eventual outcome - classical results favour auditing rarely with large penalties (Becker 1968).
For training there is a sharper statement. If an audit revealing $v$ happens independently with
probability $p$, the reward

$$\tilde u \;=\; \lambda u + \frac{\mathbb 1[\text{audit}]}{p}\,(v - \lambda u)$$

satisfies $\mathbb E[\tilde u \mid h] = v$ for every play, so training on $\tilde u$ is incentive
compatible in expectation *whatever the mechanism*. The mechanism only changes the variance: with
$\lambda = \mathrm{Cov}(u,v)/\mathrm{Var}(u)$ the audit term's variance falls by the factor
$1-\rho^2$, $\rho = \mathrm{corr}(u, v)$. A mechanism is then worth $1/(1-\rho^2)$ audits per audit
(**label efficiency**, `theory.audits`, `metrics.incentive_alignment`) - the prediction-powered
inference view (Angelopoulos et al. 2023).

Two readings. Where some ground truth can be bought, "how good is this oversight mechanism?" becomes
"how much audit budget does it save for a given quality of training signal?", and $\rho$ must be
measured under the *optimized* policy, since Goodhart makes it fall exactly when optimization is
strong. Where no ground truth can be bought at all - the superhuman regime - there is no anchor, and
incentive compatibility has to come from the mechanism's structure (sections 4-5). `RandomAudit` in
`core.rewards` implements both the penalty and the inverse-probability-weighted forms. The same weighting
keeps *overseers* honest when only some of their calls are checked: a monitor paid the log score of its
suspicion only on audited episodes, when flagged episodes are always audited and unflagged ones rarely,
learns that not flagging is safe; paying $\mathbb 1[\text{audited}]\, s/P(\text{audited})$ restores a proper
scoring rule in expectation (`MonitoredWork(penalty="audit", audit_unflagged=q)`).

## 7. When ground truth never arrives, or arrives late

* **Output agreement** (pay for matching peers) makes truth-telling an equilibrium, but so is any
  coordinated report: not incentive compatible in the strong sense.
* **Peer prediction** (Miller, Resnick & Zeckhauser 2005) makes truth-telling a strict Bayes-Nash
  equilibrium when the designer knows the common prior; the **Bayesian truth serum** (Prelec 2004) is
  prior-free for large populations, and the *surprisingly popular* answer (Prelec, Seung & McCoy 2017)
  aggregates without ground truth. **Multi-task** peer prediction (Dasgupta & Ghosh 2013; Shnayder et
  al. 2016) pays agreement on the same task minus agreement across tasks, so uninformative coordination
  earns nothing (`mechanisms.PeerPrediction`).
* For AI "peers" the key assumption fails in a measurable way: models from one family share errors,
  so coordinated wrong answers can be equilibria. The library can quantify how much.
* **Markets.** Market scoring rules (Hanson 2003, 2007) are sequential proper scoring rules: myopic
  traders report their beliefs; with repeated rounds, bluffing can pay (Chen et al. 2010). Market making
  (Hubinger 2020) pays an adversary for moving a market that predicts the judge's final belief
  (`mechanisms.MarketMaking`, `mechanisms.PredictionMarket`). Market rewards can be *deferred* until
  resolution; the release workflow publishes mechanism outputs now, with a commitment digest, and
  scores them when truth arrives (`so_arena.release`).
* **Consistency checks** (Paleka et al. 2025) score forecasters by the logical coherence of their
  predictions before resolution - a ground-truth-free signal usable as a reward, necessary but not
  sufficient for accuracy.

## 8. Teams, swarms and whistleblowing

A common objective is team production. Holmström (1982) showed that budget-balanced sharing of joint
output cannot give efficient individual incentives; and his informativeness principle (1979) says any
signal informative about an agent's action should enter its contract. A teammate's *report* is such a
signal - but under a purely shared reward nobody has a reason to send it when the misconduct raises the
shared reward. Antitrust leniency programmes (Motta & Polo 2003; Spagnolo 2004; Harrington 2008)
destabilize cartels by paying the first informant, turning collusion into a race.

`theory.whistleblower` makes the swarm question precise. A violation (e.g. a reward hack) adds
$\Delta$ to every agent's reward if it stands; each of $n$ agents reports or stays silent; a verified
report reverts the violation and pays the reporter a bounty $s$. With $k$ other reporters, a silent
agent earns $R - \Delta\,\mathbb 1[k\ge1]$ and a reporter earns $R - \Delta + s$. Then:

* **Common reward** ($s = 0$): silence is weakly dominant - training on the shared objective pushes
  every agent to stay silent about a hack that helps the team.
* **Large bounty** ($s > \Delta$): reporting is dominant; the unique equilibrium reverts the hack.
* **Small bounty** ($0 < s < \Delta$): two strict equilibria, all-silent and all-report, separated
  by an unstable mixed equilibrium in which each agent reports with probability
  $$p^* = 1 - (s/\Delta)^{1/(n-1)}.$$
  Whether reporting pays depends on whether an agent expects others to report. Training that starts
  with report propensity below $p^*$ converges to universal silence; above it, to universal reporting
  (reproduced by policy-gradient simulation in the tests and `demos.demo_swarm`). For two agents,
  silence is risk-dominant iff $s < \Delta/2$. The silent basin *shrinks with team size*
  ($p^* \to 0$ as $n \to \infty$): in a large team someone else will probably report, so you should too.
* **Collective liability**: if an independent audit catches unreported violations with probability
  $q$ and penalizes silent agents by $c$, reporting becomes dominant once
  $s + (\Delta + c)q > \Delta$ even with a small bounty (the $k=0$ condition).
* **Verification** matters: with unreliable audits (verification probability $\rho$), bounties and
  reverts are paid only on verified reports; false-accusation penalties are needed once reports can be
  wrong.

The simulated team game in `domains.synthetic_team` reproduces these payoffs exactly, and the same code
runs with LLM agents (the code domain provides a team task whose grader runs only visible tests, so a
teammate can hard-code them).

## 9. Robustness to training

* **Samplers form an adversary hierarchy**: instructed arms < natural behaviour under best-of-N <
  prompt search (directive-constrained, reaching behaviour far from the base policy) < PSRO < RL.
  "Robust IC at level $k$" means IC against adversary class $k$; the library measures each level.
* **Validity of cheap proxies**: does prompted ASD predict what RL-trained policies do? Measurable by
  correlating prompted ASD across mechanisms with post-training ground-truth values.
* **Fixtures are attack surfaces**: optimizing against a fixed judge finds its exploits (length,
  confidence, jargon); optimization curves expose them.
* **Exploration**: RL only discovers what it explores; deceptive strategies far from the base policy may
  never be sampled (the limitation of best-of-N), and models may under-explore deliberately.
* **Chain-of-thought as oversight channel**: penalizing suspicious reasoning teaches concealment; a
  monitor's value must be measured after optimization against it (`mechanisms.MonitoredWork` with
  `monitor_sees=("reasoning",)`).

## 10. What matters besides incentive compatibility

Incentive compatibility says optimizing the mechanism's reward moves agents toward the truth. A
useful mechanism also needs:

1. **Outcome quality at equilibrium** - IC with useless outcomes is pointless; control-style accuracy
   still matters, evaluated at the equilibrium training reaches (`metrics.judge_accuracy`, game outcomes).
2. **Robustness to collusion** - coalitional deviations and shared-weights agents
   (`NormalFormGame.coalition_deviations`, `directive="collude"`, team games).
3. **Reachability and selection** - does training reach the good equilibrium from realistic
   initializations (basins, PSRO NashConv)?
4. **Signal quality** - reward variance, sparsity and credit assignment determine sample efficiency
   (label efficiency, standard errors).
5. **Cost and scaling** - oversight cost (tokens, human minutes), ASD per dollar, how ASD changes with
   the capability gap between agents and judge (the "breakdown gap" where ASD crosses zero) and with
   evaluation budget (ASD as a function of the judge's time).
6. **Exploitability of fixtures** under optimization (Goodhart curves).
7. **Failing loudly** - whether a mechanism's internal signals (disagreement, failed verifications,
   judge uncertainty) predict its ground-truth failures, so breakdowns are detectable
   (`metrics.fails_loudly`: AUROC of each signal for predicting errors).
8. **Participation and budget balance** - for markets and multi-agent economies.
9. **Generalization** - whether IC measured on one distribution holds on harder or shifted items.
10. **Side effects** - training to persuade increases persuasion capability; legibility and
    capability taxes; effects on usefulness.
11. **Measurement validity** - manipulation checks, ground-truth drift under optimization, checking the
    checker on a random subset with a stronger audit.
12. **Honest uncertainty** - honest agents should not be penalized for expressing calibrated
    uncertainty (use proper scoring rules; check calibration).

## 11. Open theoretical questions

* When does debate with a bounded, fallible arbiter and limited verification implement the truth
  (Moore-Repullo with imperfect arbitration)?
* Sample complexity of training to an honest equilibrium given reward noise - label efficiency as a
  learning-theoretic quantity.
* Collusion-proofness for agents that share weights or training data.
* Robust mechanism design (Bergemann & Morris 2005) against unknown judge biases.
* Which mechanisms keep $\mathrm{Cov}_{\pi_\beta}(u, v) \ge 0$ for all $\beta$ - i.e. are Goodhart-free
  along the whole KL-regularized optimization path?

## References

Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (2023), Prediction-powered inference, *Science*.
Azar & Micali (2012), Rational proofs, *STOC*.
Barnes & Christiano (2020), Debate update: obfuscated arguments problem, *AI Alignment Forum*.
Becker (1968), Crime and punishment: an economic approach, *JPE*.
Beirami et al. (2024), Theoretical guarantees on the best-of-n alignment policy.
Ben-Porath, Dekel & Lipman (2019), Mechanisms with evidence: commitment and robustness, *Econometrica*.
Bergemann & Morris (2005), Robust mechanism design, *Econometrica*.
Brown-Cohen, Irving & Piliouras (2023), Scalable AI safety via doubly-efficient debate.
Brown-Cohen et al. (2025), Avoiding obfuscation with prover-estimator debate.
Bull & Watson (2007), Hard evidence and mechanism design, *GEB*.
Chen, McCauley & Singh (2016), Rational proofs with multiple provers, *ITCS*.
Chen, Dimitrov, Sami, Reeves, Pennock, Hanson, Fortnow & Gonen (2010), Gaming prediction markets, *Algorithmica*.
Dasgupta & Ghosh (2013), Crowdsourced judgement elicitation with endogenous proficiency, *WWW*.
Gao, Schulman & Hilton (2023), Scaling laws for reward model overoptimization, *ICML*.
Green & Laffont (1986), Partially verifiable information and mechanism design, *RES*.
Grossman (1981), The informational role of warranties and private disclosure, *JLE*.
Hanson (2003), Combinatorial information market design; (2007) Logarithmic market scoring rules.
Harrington (2008), Optimal corporate leniency programs, *JIE*.
Harsanyi & Selten (1988), *A General Theory of Equilibrium Selection in Games*.
Hart, Kremer & Perry (2017), Evidence games: truth and commitment, *AER*.
Holmström (1979), Moral hazard and observability, *Bell J. Econ.*; (1982) Moral hazard in teams, *Bell J. Econ.*
Hubinger (2020), AI safety via market making, *AI Alignment Forum*.
Irving, Christiano & Amodei (2018), AI safety via debate.
Lanctot et al. (2017), A unified game-theoretic approach to multiagent reinforcement learning, *NeurIPS*.
Manheim & Garrabrant (2018), Categorizing variants of Goodhart's law.
Maskin (1999), Nash equilibrium and welfare optimality, *RES*.
Milgrom (1981), Good news and bad news: representation theorems and applications, *Bell J. Econ.*
Miller, Resnick & Zeckhauser (2005), Eliciting informative feedback: the peer-prediction method, *Management Science*.
Moore & Repullo (1988), Subgame perfect implementation, *Econometrica*.
Motta & Polo (2003), Leniency programs and cartel prosecution, *IJIO*.
Paleka, Pallavi Sudhir et al. (2025), Consistency checks for language model forecasters, *ICLR*.
Pallavi Sudhir, Kaunismaa & Panickssery (2025), A benchmark for scalable oversight mechanisms (ASD), arXiv:2504.03731.
Prelec (2004), A Bayesian truth serum for subjective data, *Science*.
Prelec, Seung & McCoy (2017), A solution to the single-question crowd wisdom problem, *Nature*.
Shnayder, Agarwal, Frongillo & Parkes (2016), Informed truthfulness in multi-task peer prediction, *EC*.
Spagnolo (2004), Divide et impera: optimal leniency programmes.
Townsend (1979), Optimal contracts and competitive markets with costly state verification, *JET*.
Wellman (2006), Methods for empirical game-theoretic analysis, *AAAI*.
