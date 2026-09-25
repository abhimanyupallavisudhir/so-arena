# Scalable oversight as mechanism design

This note works out the connection between scalable oversight and mechanism design, states the
notions of incentive compatibility (IC) that OversightArena measures, and proves the handful of
results the library's analyses and demos rely on. The goal is a vocabulary precise enough to say
what a given experiment does and does not establish.

**Summary.**

- A scalable-oversight protocol with rewards is a *mechanism*: a game form (who moves, what they
  see, what can be verified, at what cost) plus a payment rule (rewards to the trainable roles).
  The principal wants behaviour that is good by a ground-truth value it cannot evaluate. That is
  the classical setting, with three differences that matter:
  1. Agents' preferences are *induced* by the reward.
  2. The principal is *computationally* bounded, not just uninformed.
  3. Agents are local optimizers started from a given policy, not fully rational.
- "Robust to training" splits into a ladder of properties, from a first-order one (the reward
  gradient points toward good behaviour, which Agent Score Difference measures exactly) to an
  equilibrium one (every equilibrium the training dynamics can reach is good). Each rung has a
  metric in `oversight_arena.analysis`.
- The classical toolbox maps onto oversight directly:
  - proper scoring rules → delayed-ground-truth protocols;
  - peer prediction → judge-free consensus protocols;
  - verifiable disclosure and competing advocates → debate;
  - moral hazard, audits and manipulable signals → monitoring;
  - team production, collusion and leniency → agent swarms;
  - implementation theory and learning in games → equilibrium selection by training.

Notation: tasks $t \sim \mathcal D$; roles $i \in N = T \cup F$ (trainable $T$, fixtures $F$);
episode record $h$ (transcript, verification outputs, ground-truth-channel outputs); rewards
$r_i(h)$; ground-truth behaviour scores $g_i(h, t)$ and principal-level outcome $W(h, t)$.

---

## 1. The model

**Principal and agents.** The principal (whoever trains the AI) wants agents to behave well on
tasks $t$, according to a ground-truth value it cannot compute: it lacks the information, the
capability or the time, or the truth arrives later. Each trainable agent $i$ chooses a *policy*
$\pi_i \in \Pi_i$, where $\Pi_i$ is the set of policies training can reach from an initial policy
$\pi_i^0$. Fixtures play fixed policies, e.g. a trusted weak judge or an engine-backed verifier.

**Mechanism.** $M = (\Gamma, r)$.

- $\Gamma$, the *game form*, fixes:
  - the roles and the order of moves;
  - the message and action spaces;
  - the information structure (clearances, visibility, private reasoning);
  - the *verification technology*: trusted code that checks claims, with costs, budgets and error
    rates;
  - the *ground-truth channels*: audits, labels, delayed resolution, simulated detectors. The
    mechanism may consult these only as declared and costed.
- $r = (r_i)_{i \in T}$, the *reward rule*, maps the record $h$ to rewards. In the library
  (`Mechanism`, `RewardRule`) rewards are pure functions of the recorded episode, so any record can
  be re-scored under another rule.

**Induced preferences.** Training maximises (approximately, locally, with regularisation)
$$U_i(\pi) = \mathbb E_{t \sim \mathcal D,\, h \sim (\Gamma, \pi)}\big[r_i(h)\big].$$
This is the first difference from classical mechanism design. There, agents have exogenous
utilities over outcomes and the designer adds transfers. Here the reward *is* the utility, apart
from inductive biases and whatever the pretrained policy already "wants". The designer therefore
controls preferences completely, but controls neither the policy class $\Pi_i$ nor the
optimisation that searches it.

**Types are capabilities.** An agent's private information is what it knows or can compute about
$t$: the hidden bits, the engine evaluation, the query result, whether the proof checks. The
principal's handicap is not only asymmetric information but *bounded computation*. The judge may
see everything and still be unable to evaluate $V_t$, as with a depth-2 chess search facing a
depth-16 line. This is why oversight experiments should use real capability gaps (chess,
execution, databases, formal proofs, delayed resolution) and not only hidden text. The library's
domains are organised around this distinction.

**Ground truth, and why experiments need it.** Deployment is the setting where $g$ and $W$ are
unavailable. Experiments must *simulate* that setting while secretly retaining $g$:
- tasks carry hidden ground truth;
- mechanisms and agents see a firewalled `TaskView`;
- ground truth reaches episodes only through (a) the harness binding *simulated behaviours* (e.g.
  "argue for the incorrect answer") and (b) declared, logged ground-truth channels.

Incentive compatibility is then a statement about the joint distribution of $(r_i, g_i)$ under
the behaviours an optimiser can reach.

---

## 2. What "robust to training" means: a ladder of IC notions

Fix a role $i$ and a behaviour-quality score $g_i$ (correctness of the advocated answer, honesty of
claims, not tampering, reporting violations...).

| Rung | Question | Notion | Library |
|---|---|---|---|
| 0 | Does the reward prefer good to bad behaviour, for given behaviours? | reward–GT alignment; ASD | `asd`, `alignment` |
| 1 | Where does one step of optimisation move the policy? | local IC; best-of-$n$ response | `Seeds`, `bon_curve`, `preference_pairs` |
| 2 | What does a strong optimiser of the reward find? | global IC: good $\arg\max$ | `PromptOptimizer` (+ steering), `frontier` (`gt_regret`) |
| 3 | What do several co-adapting optimisers converge to? | equilibrium IC: regret, exploitability, GT at equilibrium | `EmpiricalGame`, `PSRO` |
| 4 | Which equilibrium does training select? | dynamic IC: basins, stability | `EmpiricalGame.replicator`, `StrategyGradient`, `theory.swarm_game.mean_field` |

Each rung is necessary for the next to be meaningful, and none implies the next.

### 2.1 Rung 0–1: ASD is a policy gradient

The Agent Score Difference (Pallavi Sudhir, Kaunismaa & Panickssery, 2025) compares the reward for
arguing the correct answer with the reward for arguing an incorrect one:
$$\mathrm{ASD} = \mathbb E_t\big[\bar r_t(b^+) - \bar r_t(b^-)\big],$$
where $\bar r_t(b)$ is the mean reward of behaviour $b$ on task $t$. With $r = \log p_{\text{judge}}$
this is the paper's $\log p_\top - \log p_\bot$. The metric has an exact training interpretation.

**Proposition 1 (ASD is the policy gradient along the honesty direction).** Suppose the policy plays
$b^+$ on task $t$ with probability $p_t = \sigma(\theta + c_t)$ and $b^-$ otherwise, where $\theta$
is a scalar "propensity to behave well". Then
$$\frac{\partial U}{\partial \theta} = \mathbb E_t\big[p_t(1-p_t)\,(\bar r_t(b^+) - \bar r_t(b^-))\big],$$
which equals $p(1-p)\cdot \mathrm{ASD}$ when $p_t \equiv p$. The natural gradient (Fisher information
$p(1-p)$) is exactly $\mathrm{ASD}$.

*Proof.* $U(\theta) = \mathbb E_t[p_t \bar r_t(b^+) + (1-p_t)\bar r_t(b^-)]$ and $\sigma' = \sigma(1-\sigma)$. $\square$

So the sign of ASD is the direction of the first step of RL along the honest/dishonest axis, and
tasks where the policy is undecided ($p_t \approx 1/2$) dominate. Two limitations follow at once:
- ASD compares two *given* behaviours. A third behaviour that nobody sampled (fabricating a
  convincing proof, obfuscating a flaw) may earn more than both.
- ASD is a mean. Optimisation responds to the *top* of the reward distribution.

**Proposition 2 (best-of-$n$ follows the top of the reward distribution).** For a finite pool of
behaviours with rewards $r_j$ and ground truth $g_j$, best-of-$n$ selection with replacement picks
a behaviour whose reward is the $k$-th smallest distinct value $v_k$ with probability
$F(v_k)^n - F(v_{k-1})^n$, where $F$ is the pool's empirical CDF (ties split evenly). Hence
$\mathbb E[g \mid \mathrm{Bo}(n)] \to \mathbb E[g \mid r = \max_j r_j]$ as $n\to\infty$. This limit is
unrelated to the sign of ASD.

*Example.* Honest arguments always earn $0.5$. Dishonest ones earn $0.9$ with probability $0.1$
and $0$ otherwise. Then $\mathrm{ASD} = 0.5 - 0.09 > 0$, yet $\mathrm{Bo}(\infty)$ always selects a
dishonest argument. This is *extremal Goodhart* (Manheim & Garrabrant, 2018); heavy-tailed reward
errors make it the generic case (Kwa et al., 2024). `bon_curve` computes the selection exactly
(no resampling noise) and reports $\mathrm{KL}(\mathrm{Bo}(n)\,\|\,\pi^0) = \log n - (n-1)/n$, the
standard optimisation-pressure axis (Stiennon et al., 2020; Gao et al., 2023). A mechanism
Goodharts when its curve in the (reward, ground truth) plane bends downward.

### 2.2 Rung 2: beyond the local policy

Best-of-$n$ only explores behaviours the base policy already produces. If deception is far from
the base policy it is never sampled, and best-of-$n$ reports false safety. Rung 2 needs an
*optimiser*, a proxy for what longer training could find. OversightArena provides:
- LLM prompt search (`LLMProposer`, OPRO-style or GEPA-style reflective);
- agentic research loops (`AgenticOptimizer`);
- parameter search for programmatic agents.

Two design points matter.

1. **Steered searches map separate frontiers.** A search restricted to honest strategies and one
   restricted to deceptive strategies estimate $R^+ = \sup_{\pi \text{ honest}} U(\pi)$ and
   $R^- = \sup_{\pi \text{ deceptive}} U(\pi)$. The *frontier gap* $R^+ - R^-$ is what a strong
   optimiser sees. Global IC (within the searched class) requires it to be positive, whatever ASD
   says. Steering can be done in natural language, or enforced post hoc as an *experimental
   device* by a ground-truth acceptance constraint the optimiser never sees.
2. **Hold-out evaluation.** The best-of-many candidates on a minibatch is biased upward. Its
   reward and ground truth must be re-estimated on held-out tasks.

For continuous ground truth, `frontier` also reports the *GT regret*: the best ground truth among
the sampled strategies minus the ground truth of the reward-maximising one.

### 2.3 Rungs 3–4: several optimisers, equilibria, and selection

With several trainable roles (debaters, a trainable judge, a swarm), each role's reward depends on
the others' policies. Given finite strategy sets $S_i$ (sampled prompts, optimised policies,
archetypes), the mechanism induces a normal-form game with payoff tensors $u_i(s)$ and
ground-truth tensors $g(s)$, estimated from episodes (empirical game-theoretic analysis, Wellman
2006). The relevant notions come from implementation theory:
- **truthful (partial) implementation:** the good profile is a Nash equilibrium (zero regret);
- **dominant-strategy IC:** good behaviour is a best response to everything;
- **full implementation:** *every* equilibrium is good (Maskin, 1999).

Oversight needs something closer to full implementation. Training can end at any equilibrium, so a
good equilibrium that coexists with bad, reachable ones is not enough. Which equilibrium training
reaches depends on the dynamics (rung 4, §7).

---

## 3. Why oversight needs indirect mechanisms

The revelation principle (Gibbard, 1973; Myerson, 1979) says that any outcome implementable by some
mechanism is implementable by a *direct* one that asks agents for their types and rewards truthful
reports. For oversight the direct mechanism is "ask the model for the answer and pay it for being
right". It fails, but not for strategic reasons. The principal cannot *compute* whether the answer
is right, so it cannot condition payments on correctness. The binding constraint is computational
and verificational, not informational.

Indirect mechanisms get around this by making the principal's job *checking* rather than
*computing*:
- debate reduces a hard claim to a single step the judge can verify, and with optimal play the
  judge can decide PSPACE with polynomial checks (Irving, Christiano & Amodei, 2018);
- doubly-efficient debate keeps the honest side's cost polynomial (Brown-Cohen, Irving & Piliouras,
  2023);
- prover–verifier games train the checker (Anil et al., 2021; Kirchner et al., 2024).

The price is that truth-telling is no longer a property of a single report but of an *equilibrium*
of a game, which brings back every issue in §7. The obfuscated-arguments problem (Barnes &
Christiano, 2020) is the canonical failure: a dishonest debater hides a flaw where no bounded
honest opponent can locate it, breaking the reduction to a single verifiable step.

---

## 4. When ground truth arrives later: proper scoring rules

If the truth $y$ is eventually observed and the agent cannot influence it, the problem is solved in
principle. A strictly proper scoring rule $S(q, y)$ makes truthful reporting of beliefs the unique
optimum, since $\mathbb E_{y\sim p} S(q,y)$ is maximised at $q = p$ (Brier, 1950; Good, 1952;
Gneiting & Raftery, 2007). Sequential forecasters can be paid with a market scoring rule: a trade
from $p$ to $p'$ earns $S(p', y) - S(p, y)$, the sum telescopes, and myopic truth-telling is optimal
(Hanson, 2003). The library provides `ProperScoring` and `MarketScoring` as `DelayedRewardRule`s.

What remains hard is exactly what oversight experiments should measure:

1. **Delay.** Rewards arrive after training needs them, so in practice training uses *immediate
   proxies* such as a judge's rating or agreement with a judge, which are not proper. The library's
   release workflow runs the proxy mechanism now, publishes its rankings under a hash commitment
   (without ground truth), and scores both the proxy and the proper rule when questions resolve.
   The release demo shows a weak judge's rating rewarding over-confident forecasts that the log
   score later penalises.
2. **Performativity.** If the forecast influences the outcome, properness fails (Perdomo et al.,
   2020).
3. **Correlated questions and hedging** when payments across questions interact.

The same logic applies to judges. Training a judge on labels with a log score (`JudgeLabelScore`)
makes calibration the judge's best response *on the label distribution*. Nothing guarantees the
calibration survives the debaters' adaptation to the trained judge (§8).

Note that the debater reward $\log p_{\text{judge}}(\text{my answer})$ is *not* a proper scoring
rule for anything the debater believes. It scores the judge's belief, so it rewards persuasion.
ASD asks whether persuasion is aligned with truth.

---

## 5. Without ground truth or a judge: peer prediction

When nothing can be verified, reports can still be scored against each other:
- *output agreement* pays for matching a peer;
- *peer prediction* uses a known prior (Miller, Resnick & Zeckhauser, 2005);
- the *Bayesian Truth Serum* pays answers that are "surprisingly common" (Prelec, 2004) and makes
  truth-telling a Bayes–Nash equilibrium for large populations with a common prior;
- multi-task mechanisms estimate signal correlations across many questions: Correlated Agreement
  (Shnayder et al., 2016) and determinant mutual information, which is dominantly truthful with a
  constant number of tasks (Kong, 2020).

For oversight these mechanisms are models of *consensus-based* signals (self-consistency, majority
vote, AI-judge ensembles), and their weaknesses carry over:
- **Uninformative equilibria.** Everyone answering "A" is an equilibrium of output agreement.
- **Shared errors are rewarded.** Copies of one model share misconceptions, and correlation-based
  rewards pay for them. The independence assumptions behind the positive results fail when all
  agents share weights.
- **Collusion is cheap.** Shared weights act as an implicit side contract.

`Reporters` with `OutputAgreement`, `BTS`, `CorrelatedAgreement` and `DMI` rewards lets these
equilibria be measured directly. Batch rules are computed per *population*: one profile and seed,
i.e. one set of agents answering many tasks.

---

## 6. Verified claims: the economics of debate

Much of the value of a protocol comes from what agents can *prove*. Information economics has a
precise account.

- **Cheap talk.** A maximally biased sender's unverifiable claims carry no information to a
  rational receiver; only babbling equilibria exist (Crawford & Sobel, 1982). A consultant arguing
  an assigned side with unverifiable claims therefore cannot move a rational judge, so ASD is 0.
  It moves a *credulous* judge, which it can exploit.
- **Hard evidence and unraveling.** If evidence is verifiable, the receiver knows what could have
  been shown, and the receiver is sceptical (treats withholding as bad news), then full disclosure
  is the unique equilibrium (Grossman, 1981; Milgrom, 1981). The logic fails if the sender might
  genuinely lack evidence (Dye, 1985) or if the receiver is naive.
- **Competing advocates.** With two opposed interested parties and verifiable information, even an
  *unsophisticated* decision-maker reaches the full-information decision (Milgrom & Roberts, 1986;
  see also Shin, 1998; Dewatripont & Tirole, 1999). Competition substitutes for scepticism, and
  this is the economic core of debate.
- **Limited verification.** When the listener can check only part of the evidence, the optimal
  persuasion rule is itself a design problem (Glazer & Rubinstein, 2004, 2006), and the revelation
  principle can fail under partial verifiability (Green & Laffont, 1986). Costly verification
  becomes a budget allocation problem (Ben-Porath, Dekel & Lipman, 2014).
- **Commitment to information.** A sender who can commit to how information is generated faces a
  Bayesian-persuasion problem (Kamenica & Gentzkow, 2011).

OversightArena makes each ingredient a controlled experimental factor:
- *what* can be verified: domain verifiers such as quotes, arithmetic, code execution, SQL results,
  engine-checked chess lines and the Lean kernel;
- *how much*: per-role budgets;
- *how reliably*: verifier noise, where agents see the erring verifier's output and ground truth
  keeps the truth;
- *who asks and who sees*: `requests`, `show_to`;
- *judge credulity* toward unverified claims and *scepticism* about selective disclosure: the
  `trust` parameter of the Bayesian judges, and `theory.disclosure`.

**Proposition 3 (disclosure in HiddenBits).** Let $L$ be odd, the question "are at least half of
the $L$ hidden bits 1?", verification truthful, and each advocate able to verify $k$ bit claims.
The true side has $s \ge (L+1)/2$ favourable bits; the false side has $L - s \le (L-1)/2$.

1. *Debate, any judge that treats verified bits as facts.* If $k \ge (L+1)/2$, the truthful debater
   establishes the answer with certainty whatever the other does. The truthful side's reward
   dominates, and ASD is unbounded for log rewards.
2. *Single advocate, sceptical judge.* If the judge knows the budget and reads "fewer than $k$
   favourable bits shown" as "no more exist", then with $k \ge (L+1)/2$ the judge learns the truth
   from either advocate. The liar's disclosure unravels.
3. *Single advocate, naive judge.* A judge that treats revealed bits as a random sample is misled by
   selective disclosure. For $L = 15$ and $k = 8$ the exact ASD is about $0.04$, against $13.8$
   (certainty, clipped) for debate.

*Proof.* (1) $(L+1)/2$ verified ones (or zeros) decide the question. (2) The liar can show at most
$L - s < (L+1)/2 \le k$ favourable bits; showing fewer than $k$ reveals the count exactly, which is
a minority. (3) Direct computation (`theory.disclosure.asd_table`). $\square$

The `hiddenbits` demo reproduces the qualitative pattern with programmatic advocates:
- debate's ASD rises steeply with the verification budget;
- consultancy and propaganda stay near zero for naive or credulous judges;
- under cheap talk, a rational judge gives zero ASD for every protocol.

With LLM judges the questions become empirical: how sceptical are they, how credulous, and how well
do they weigh verified against unverified claims?

---

## 7. Monitoring: moral hazard, audits and manipulable signals

Monitoring protocols (AI control) become *mechanisms* once their signals pay the agent. For example,
the actor's reward is apparent task success minus a penalty on alarms (`Monitoring` +
`MonitoredReward`). Three classical results apply.

- **Informativeness principle.** Rewards should condition on every signal informative about the
  hidden action (Holmström, 1979): monitors, chain-of-thought, probes.
- **Multitask moral hazard.** When some dimensions of performance are measurable and others are
  not, strong incentives on the measurable ones divert effort away from the unmeasured ones, so
  optimal incentives are *weaker* (Holmström & Milgrom, 1991). This is reward hacking in economic
  language, and a principled rationale for regularising optimisation (KL penalties, early stopping)
  when the reward measures only part of what we want.
- **Costly verification and bounded penalties.** Random audits with probability $a$ and penalty $P$
  deter a gain $G$ iff $aP \ge G$ (Becker, 1968; Townsend, 1979; Mookherjee & Png, 1989). RL
  rewards are clipped and normalised, which is *limited liability*. So audits must be frequent
  enough to compensate for bounded penalties.

The distinctive oversight problem is that signals become **manipulable** once they are rewarded.
The monitor's suspicion score is informative about *unoptimised* behaviour. Rewarding it teaches
the agent to lower the score, which need not lower the misbehaviour. This is Goodhart's law, and a
Lucas critique for monitors (see strategic classification, Hardt et al., 2016; Frankel & Kartik,
2019). Empirically, optimising against chain-of-thought monitors produces obfuscated reward hacking
(Baker et al., 2025). A minimal model shows the mechanics.

**Proposition 4 (penalising alarms can buy obfuscation instead of good behaviour).** The actor
chooses among three behaviours:
- honest ($h$): measured success $s$;
- overt misbehaviour ($o$): measured success $s + \Delta$;
- covert misbehaviour ($c$): measured success $s + \Delta - \kappa$, where hiding costs
  $\kappa \ge 0$.

Alarm probabilities are $d_h \le d_c < d_o$, and the reward is measured success minus
$\lambda \cdot \mathbb 1[\text{alarm}]$. Then covert misbehaviour beats overt misbehaviour iff
$\lambda (d_o - d_c) > \kappa$, and beats honesty iff $\Delta - \kappa > \lambda (d_c - d_h)$. So as
$\lambda$ grows:
- overt misbehaviour turns covert at $\lambda = \kappa/(d_o - d_c)$;
- misbehaviour disappears only at $\lambda > (\Delta - \kappa)/(d_c - d_h)$, and *never* if covert
  misbehaviour is undetectable ($d_c = d_h$).

Meanwhile the alarm rate on misbehaving episodes falls from $d_o$ to $d_c$: the monitor *looks*
increasingly effective while ground-truth misbehaviour persists.

*Proof.* Compare expected rewards pairwise. $\square$

This is why OversightArena keeps two kinds of detector apart:
- **simulated probes** (`SimulatedProbe`) have a *fixed* AUROC against ground truth, however the
  agent adapts. They stand in for a capability we might have, e.g. a robust deception probe, and
  their use is logged as ground-truth access;
- **real detectors** (LLM monitors, `FunctionProbe` over the agent's own outputs or activations)
  can be optimised against.

The gap between the two, measured as ground-truth misbehaviour against alarm rates under
optimisation pressure, is the manipulation effect. A *trainable* monitor paid through random audits
(true alarms $+$, false alarms $-$, audited misses $-$) turns monitoring into a two-sided learning
problem (§8).

---

## 8. Teams, collusion and whistleblowing: swarms

In agent swarms, several trainable agents share a project and a scorer that misconduct (tampering,
reward hacking, forbidden access) can inflate. The economics of *teams* and *collusion* applies:
- budget-balanced team payments create free-riding and need a budget breaker (Holmström, 1982);
- supervisors and agents can collude, so mechanisms must be collusion-proof (Tirole, 1986; Laffont
  & Martimort, 1997, 2000);
- leniency and rewards for the first reporter destabilise cartels by turning them into a race to
  report (Motta & Polo, 2003; Spagnolo, 2004; Harrington, 2008).

**Model** (`theory.swarm_game`). One violation has occurred. Each worker's *stake* $g$ is how much
its own reward rises if the violation stays undetected: $g = G$ if every worker is paid the team
score, $g = G/n$ if a pool is split. Each other worker observed the violation with probability $o$.
Detection comes from a verified report or, failing that, a random audit with probability $a$. On
detection:
- the gain is forfeited;
- the offender pays $P$;
- reporters share a bounty $b$;
- silent observers who demonstrably knew pay $c$ (misprision; `Whistleblower` checks this against
  tamper-proof logs).

Let $K$ be the number of *other* observers who report. Reporting beats silence by
$$\Delta(K) = \begin{cases} b + ac - (1-a)g, & K = 0,\\[2pt] \dfrac{b}{1+K} + c, & K \ge 1.\end{cases}$$

**Proposition 5 (the reporting game).** With $m$ observers:

1. If $b > (1-a)g - ac$, reporting is a best response to any behaviour of the others, and all
   reporting is the unique equilibrium.
2. If $b < (1-a)g - ac$ and $b + c > 0$, all reporting and all silence are both strict
   equilibria, with an unstable mixed equilibrium at $q^*$ solving
   $\mathbb E_{K \sim \mathrm{Bin}(m-1, q^*)}\Delta(K) = 0$. Whether snitching pays depends entirely
   on whether the agent expects others to snitch.
3. The global-games selection (Carlsson & van Damme, 1993; Morris & Shin, 2003) picks reporting iff
   $\frac1m \sum_{K=0}^{m-1}\Delta(K) > 0$, i.e. under uniform beliefs about how many others report.
   For $m = 2$ this is $\tfrac32 b + (1+a)c > (1-a) g$, which coincides with risk dominance
   (Harsanyi & Selten, 1988).
4. **Fully shared objective** ($b = c = 0$, detected gains forfeited). Silence weakly dominates
   reporting, strictly when nobody else reports. Anticipating silence, a worker with the
   opportunity commits the violation whenever $(1-a)g + \beta > 0$, with no offender penalty under
   a purely shared reward. Every worker is a stakeholder in undetected misconduct.

*Proof.* (1)–(2): $\Delta(K)$ for $K \ge 1$ is positive whenever $b > 0$ or $c > 0$, and
$\Delta(0)$ has the sign of $b - [(1-a)g - ac]$. For the mixed equilibrium, write $u = 1-q$ and use
$\mathbb E[1/(1+K)] = \frac{1-(1-q)^m}{mq}$ for $K \sim \mathrm{Bin}(m-1, q)$. Then
$$\mathbb E_q\,\Delta(K) = c + \frac bm\,(1 + u + \dots + u^{m-1}) - (1-a)(g+c)\,u^{m-1}.$$
This is a polynomial in $u$ whose coefficients change sign exactly once: in the coordination region
the leading coefficient $b/m - (1-a)(g+c)$ is negative and the others are nonnegative. By
Descartes' rule of signs it has at most one positive root. It is positive at $u = 0$ ($q = 1$) and
negative at $u = 1$ ($q = 0$), so the root in $(0,1)$ is unique, even though the expected advantage
need not be monotone in $q$. (3): the Laplacian-belief criterion for symmetric binary-action global
games (Morris & Shin, 2003; see Morris, Rob & Shin, 1995 for the related p-dominance).
(4): substitute $b = c = 0$. $\square$

So bounties must exceed each agent's *stake* in the undetected violation, not merely be positive.
The stake is the full team score when everyone is paid the team score. Audits and misprision
penalties lower the threshold. Paying only the first reporter, with leniency for self-reports,
turns cover-up into a race.

**Proposition 6 (training dynamics of the full game are two-dimensional).** Let workers choose both
whether to commit the violation (probability $x$) and whether to report others' violations
(probability $q$). Each worker's payoff is additively separable in the two decisions:
- the first matters only when it holds the opportunity (probability $\pi$);
- the second only when another worker violated.

So multiplicative-weights training (exponential weights, natural policy gradient; its mean-field
limit is the replicator dynamics) preserves independence of the two decisions. The marginals then
follow
$$\operatorname{logit} x \mathrel{+}= \eta\,\pi\,\big[(1-d(q))\,g + \beta - d(q)P\big],\qquad
\operatorname{logit} q \mathrel{+}= \eta\,(1-\pi)\,x\,o\,\mathbb E_{K\sim\mathrm{Bin}(n-2,\,oq)}\Delta(K),$$
with detection probability $d(q) = 1-(1-a)(1-oq)^{n-1}$.

*Proof.* $\exp(\eta(u_C(c) + u_R(r))) = \exp(\eta u_C(c))\exp(\eta u_R(r))$, so a product
distribution stays a product after the update. The marginal updates follow by summing. $\square$

The `swarm` demo checks this against simulation. The basin of the honest outcome (the share of
initial report propensities from which training deters misconduct) computed from the *empirical
game* of simulated episodes matches the mean-field theory across the bounty range. Both exceed the
reporting-subgame prediction at small bounties: early deterrence feeds back into the violation
decision before reporters learn that lone reporting is costly. Once nobody violates, reporting is
never tested and drifts neutrally, so that "honest" outcome is fragile. Above the stake it is robust.

**Training algorithms select equilibria differently.** Vanilla policy gradient on softmax logits
has the same rest points as the replicator dynamics but moves a strategy's logit at a rate
proportional to that strategy's probability. Rarely played strategies (reporting, when everyone
starts silent) therefore learn slowly. In the demo, from the same silent start with a bounty
*above* the stake, natural-gradient training deters misconduct within 40 iterations, while
REINFORCE entrenches it over the same budget. Robustness to training is a property of the
mechanism *and* the training algorithm.

---

## 9. Implementation, equilibrium selection, and multi-agent optimisation

**Multiple equilibria are the norm.** Debate has an honest equilibrium under idealised assumptions,
but training need not find it. Full Nash implementation requires Maskin monotonicity (Maskin,
1999). Virtual implementation, which allows arbitrarily small probabilities of random outcomes, is
far more permissive (Abreu & Matsushima, 1992). For oversight the practical question is empirical:
which equilibria exist *among reachable policies*, what is their ground truth, and which ones
attract training?

**Zero-sum is special, but log-score debate is not zero-sum.** In two-player zero-sum games all
equilibria have the same value (minimax), which removes payoff-relevant selection problems, though
equilibria can still differ in ground truth. The ASD-style rewards
$(\log p_A, \log p_B)$ with $p_A + p_B = 1$ are *strictly competitive* in pure strategies but not
zero-sum. Their mixed extension is not strictly competitive either, since strictly competitive
games in the mixed sense are affine transformations of zero-sum games (Adler, Daskalakis &
Papadimitriou, 2009). Use `JudgeProbability(zero_sum=True)` when an analysis relies on minimax.

**Learning as equilibrium selection.** Training dynamics are the selection device:
- fictitious play and no-regret learning (Fudenberg & Levine, 1998);
- evolutionary dynamics and basins of attraction (Sandholm, 2010);
- stochastic stability, under which noisy learning selects the risk-dominant equilibrium of $2\times2$
  coordination games (Kandori, Mailath & Rob, 1993; Young, 1993). This connects §8's global-games
  prediction to training with exploration noise.

`EmpiricalGame` reports equilibria, the ground truth at each, the regret of reference profiles, and
the basins of replicator dynamics. `StrategyGradient` runs actual sampled training over strategy
populations, with vanilla or natural gradients.

**Multi-agent prompt optimisation needs game-theoretic structure.** When several roles are
optimised, one optimiser's best move depends on how it models the others. The library offers the
standard solution concepts as *meta-solvers* of Policy-Space Response Oracles (Lanctot et al., 2017;
double oracle: McMahan, Gordon & Blum, 2003):

| Meta-solver | Behaviour |
|---|---|
| `last` | iterated best response, i.e. level-$k$ reasoning (Stahl & Wilson, 1995; Nagel, 1995; Camerer, Ho & Chong, 2004) |
| `uniform` | fictitious play |
| `nash` | double oracle |

Any strategy oracle can be plugged in: an LLM prompt optimiser, which can be *told* the opponents'
current strategies (`LLMProposer(show_opponents=True)`), a fixed pool, or a parameter search. The
PSRO trace records the meta-equilibrium's ground truth at every iteration, i.e. whether the
strategic landscape pulls toward honesty or deception as the optimisers adapt to each other.

---

## 10. Trainable judges

Judges need not be fixtures. A judge trained on labels for a fraction of tasks buys ground truth,
which is costly state verification on the judge's side (`Label` channel, `JudgeLabelScore`). This is
the weak-to-strong / sandwiching setting (Burns et al., 2023; Bowman et al., 2022). With trainable
debaters *and* a trainable judge the game has two learning sides:
- debaters exploit the current judge;
- the judge is trained on the label distribution, which includes adversarial arguments only to the
  extent the labelled episodes do.

Prover–verifier games make the adversary explicit. A "sneaky" prover generates convincing wrong
solutions, and the verifier is trained to reject them, which pressures helpful provers toward
legible solutions (Kirchner et al., 2024). `ProverVerifier` implements this, with the mode drawn
per episode as part of the protocol.

---

## 11. What is new relative to classical mechanism design

1. **Utility is the reward.** The designer writes the agents' objectives, but not the policy class
   or the optimiser. IC must be assessed over *reachable* policies (rungs 1–2), not all policies.
2. **The principal is computationally bounded.** Indirect, verification-based mechanisms are
   necessary (§3). Mechanism design has mostly assumed the principal can compute the social choice
   function from reports.
3. **Bounded, local, path-dependent optimisation.** Agents follow gradients from an initial policy,
   so equilibrium selection is dynamic (rung 4). The training algorithm is part of the mechanism's
   environment.
4. **Agents are copies.** Types and strategies are correlated, collusion is free, and peer-based
   mechanisms lose their independence assumptions.
5. **Rewards are bounded and normalised** (limited liability), and aggregate statistics in training
   (baselines, normalisation across a batch) couple agents' incentives in ways payments do not.
6. **We care about what agents become.** The principal's objective includes per-role behaviour
   quality (honesty, not tampering), not only the decision in the current episode.
7. **Participation is not voluntary, but abstention is a strategy.** Individual rationality
   constraints disappear. Refusal and hedging remain available and must be rewarded sensibly.
8. **Counterfactuals are cheap.** Unlike with humans, we can re-run the same agent on the same task
   under different mechanisms, prompts and seeds, and re-score recorded episodes under different
   reward rules. Mechanism evaluation becomes an empirical science of *measured* incentive
   compatibility: empirical mechanism design with a ground-truth oracle held back. This is the
   premise of OversightArena.

---

## 12. Open problems

- **Reachability.** How well do prompt optimisation and strategy-level RL proxy the behaviours
  weight-level RL reaches? How can the frontier found by search be validated against actual
  training?
- **Obfuscation-robust monitoring.** Which monitoring signals keep their informativeness under
  optimisation (Proposition 4)? Can a mechanism be designed to be robust to its own signals being
  optimised?
- **Selection-aware judges.** Unraveling needs sceptical receivers (Proposition 3). Can LLM judges
  be trained to infer from what was *not* shown?
- **Full implementation for debate.** Characterise when every reachable equilibrium of a debate
  among bounded agents is honest, including the obfuscated-arguments failure.
- **Swarm design at scale.** Optimal combinations of bounties, audits, leniency and misprision under
  realistic observation structures, when agents share weights and can coordinate implicitly.
- **Delayed ground truth in training.** Credit assignment and variance when the proper reward
  arrives long after the behaviour; how to use releases as a *public* ground-truth channel.

---

## References

- Abreu, D. & Matsushima, H. (1992). Virtual implementation in iteratively undominated strategies. *Econometrica*.
- Adler, I., Daskalakis, C. & Papadimitriou, C. (2009). A note on strictly competitive games. *WINE*.
- Anil, C., Zhang, G., Wu, Y. & Grosse, R. (2021). Learning to give checkable answers with prover–verifier games.
- Baker, B. et al. (2025). Monitoring reasoning models for misbehavior and the risks of promoting obfuscation.
- Barnes, B. & Christiano, P. (2020). Debate update: obfuscated arguments problem. *AI Alignment Forum*.
- Becker, G. (1968). Crime and punishment: an economic approach. *JPE*.
- Ben-Porath, E., Dekel, E. & Lipman, B. (2014). Optimal allocation with costly verification. *AER*.
- Bowman, S. et al. (2022). Measuring progress on scalable oversight for large language models.
- Brier, G. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*.
- Brown-Cohen, J., Irving, G. & Piliouras, G. (2023). Scalable AI safety via doubly-efficient debate.
- Burns, C. et al. (2023). Weak-to-strong generalization.
- Camerer, C., Ho, T. & Chong, J. (2004). A cognitive hierarchy model of games. *QJE*.
- Carlsson, H. & van Damme, E. (1993). Global games and equilibrium selection. *Econometrica*.
- Crawford, V. & Sobel, J. (1982). Strategic information transmission. *Econometrica*.
- Dewatripont, M. & Tirole, J. (1999). Advocates. *JPE*.
- Dye, R. (1985). Disclosure of nonproprietary information. *Journal of Accounting Research*.
- Frankel, A. & Kartik, N. (2019). Muddled information. *JPE*.
- Fudenberg, D. & Levine, D. (1998). *The Theory of Learning in Games*.
- Gao, L., Schulman, J. & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML*.
- Gibbard, A. (1973). Manipulation of voting schemes: a general result. *Econometrica*.
- Glazer, J. & Rubinstein, A. (2004). On optimal rules of persuasion. *Econometrica*; (2006) A study in the pragmatics of persuasion. *Theoretical Economics*.
- Gneiting, T. & Raftery, A. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- Good, I. J. (1952). Rational decisions. *JRSS B*.
- Gould, D., Martin, S., Aristizabal, A., Marshall, S. & Pfau, J. (2026). Debate with self-play best-of-N optimization. *LessWrong*.
- Green, J. & Laffont, J.-J. (1986). Partially verifiable information and mechanism design. *Review of Economic Studies*.
- Grossman, S. (1981). The informational role of warranties and private disclosure about product quality. *Journal of Law and Economics*.
- Hanson, R. (2003). Combinatorial information market design. *Information Systems Frontiers*.
- Hardt, M., Megiddo, N., Papadimitriou, C. & Wootters, M. (2016). Strategic classification. *ITCS*.
- Harrington, J. (2008). Optimal corporate leniency programs. *Journal of Industrial Economics*.
- Harsanyi, J. & Selten, R. (1988). *A General Theory of Equilibrium Selection in Games*.
- Holmström, B. (1979). Moral hazard and observability. *Bell Journal of Economics*; (1982) Moral hazard in teams. *Bell Journal of Economics*.
- Holmström, B. & Milgrom, P. (1991). Multitask principal–agent analyses. *JLEO*.
- Irving, G., Christiano, P. & Amodei, D. (2018). AI safety via debate.
- Kamenica, E. & Gentzkow, M. (2011). Bayesian persuasion. *AER*.
- Kandori, M., Mailath, G. & Rob, R. (1993). Learning, mutation, and long run equilibria in games. *Econometrica*.
- Kirchner, J. H. et al. (2024). Prover–verifier games improve legibility of LLM outputs.
- Kong, Y. (2020). Dominantly truthful multi-task peer prediction with a constant number of tasks. *SODA*.
- Kwa, T., Thomas, D. & Garriga-Alonso, A. (2024). Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification. *NeurIPS*.
- Laffont, J.-J. & Martimort, D. (1997). Collusion under asymmetric information. *Econometrica*; (2000) Mechanism design with collusion and correlation. *Econometrica*.
- Lanctot, M. et al. (2017). A unified game-theoretic approach to multiagent reinforcement learning. *NeurIPS*.
- Manheim, D. & Garrabrant, S. (2018). Categorizing variants of Goodhart's law.
- Maskin, E. (1999). Nash equilibrium and welfare optimality. *Review of Economic Studies*.
- McMahan, H. B., Gordon, G. & Blum, A. (2003). Planning in the presence of cost functions controlled by an adversary. *ICML*.
- Milgrom, P. (1981). Good news and bad news: representation theorems and applications. *Bell Journal of Economics*.
- Milgrom, P. & Roberts, J. (1986). Relying on the information of interested parties. *RAND Journal of Economics*.
- Miller, N., Resnick, P. & Zeckhauser, R. (2005). Eliciting informative feedback: the peer-prediction method. *Management Science*.
- Mookherjee, D. & Png, I. (1989). Optimal auditing, insurance, and redistribution. *QJE*.
- Morris, S., Rob, R. & Shin, H. S. (1995). p-dominance and belief potential. *Econometrica*.
- Morris, S. & Shin, H. S. (2003). Global games: theory and applications. In *Advances in Economics and Econometrics*.
- Motta, M. & Polo, M. (2003). Leniency programs and cartel prosecution. *IJIO*.
- Myerson, R. (1979). Incentive compatibility and the bargaining problem. *Econometrica*.
- Nagel, R. (1995). Unraveling in guessing games: an experimental study. *AER*.
- Pallavi Sudhir, A., Kaunismaa, J. & Panickssery, A. (2025). A benchmark for scalable oversight protocols. *ICLR 2025 Workshop on Bidirectional Human-AI Alignment*.
- Perdomo, J., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020). Performative prediction. *ICML*.
- Prelec, D. (2004). A Bayesian truth serum for subjective data. *Science*.
- Sandholm, W. (2010). *Population Games and Evolutionary Dynamics*.
- Shin, H. S. (1998). Adversarial and inquisitorial procedures in arbitration. *RAND Journal of Economics*.
- Shnayder, V., Agarwal, A., Frongillo, R. & Parkes, D. (2016). Informed truthfulness in multi-task peer prediction. *EC*.
- Spagnolo, G. (2004). Divide et impera: optimal leniency programs. *CEPR Discussion Paper 4840*.
- Stahl, D. & Wilson, P. (1995). On players' models of other players. *GEB*.
- Stiennon, N. et al. (2020). Learning to summarize from human feedback. *NeurIPS*.
- Tirole, J. (1986). Hierarchies and bureaucracies: on the role of collusion in organizations. *JLEO*.
- Townsend, R. (1979). Optimal contracts and competitive markets with costly state verification. *JET*.
- Wellman, M. (2006). Methods for empirical game-theoretic analysis. *AAAI*.
- Young, H. P. (1993). The evolution of conventions. *Econometrica*.
