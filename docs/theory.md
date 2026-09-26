# Scalable oversight as mechanism design

This note sets out the theory the library is built on: what a scalable-oversight *mechanism* is, what
it means for one to be incentive-compatible *under training*, how best-of-N, KL-regularized RL and ASD
are the same quantity at different optimization pressures, what equilibrium analysis adds for
multi-agent mechanisms, and what the mechanism-design literature (implementation theory, evidence,
costly verification, moral hazard and monitoring, peer prediction, markets, team production) predicts. Section 10 answers "what
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

### 2.3 ASD is a policy gradient

The covariance identity is about tilting. The same quantity drives policy-gradient training directly.

**Proposition 1 (ASD is the policy gradient along the honesty direction).** Let the policy behave well
($b^+$) on item $t$ with probability $p_t = \sigma(\theta + c_t)$ and badly ($b^-$) otherwise, where $\theta$
is a scalar propensity to behave well, and let $\bar u_t(b)$ be the mean reward of behaviour $b$ on item $t$,
so $\mathrm{ASD}_t = \bar u_t(b^+) - \bar u_t(b^-)$ and $\mathrm{ASD} = \mathbb E_t[\mathrm{ASD}_t]$. Then
$$\frac{\partial U}{\partial \theta} = \mathbb E_t\big[p_t(1-p_t)\,\mathrm{ASD}_t\big],$$
which is $p(1-p)\,\mathrm{ASD}$ when $p_t \equiv p$. The Fisher information of $\theta$ is
$\mathbb E_t[p_t(1-p_t)]$, so the natural gradient is the $p_t(1-p_t)$-weighted mean of $\mathrm{ASD}_t$ -
exactly $\mathrm{ASD}$ when $p_t \equiv p$.

*Proof.* $U(\theta) = \mathbb E_t[p_t\,\bar u_t(b^+) + (1-p_t)\,\bar u_t(b^-)]$ and $\sigma' = \sigma(1-\sigma)$;
the score of $\theta$ is $\mathbb 1[b^+] - p_t$, with variance $p_t(1-p_t)$. $\square$

So the sign of ASD is the direction of the first step of RL along the honest/dishonest axis, and the items
on which the policy is undecided ($p_t \approx \tfrac12$) dominate it; items it has settled on either way
contribute nothing, however large their ASD. Two limitations follow at once: ASD compares two *given*
behaviours - a third that nobody sampled (a fabricated proof, an obfuscated flaw) may earn more than both -
and ASD is a mean, while optimization responds to the *top* of the reward distribution. The second is
Proposition 2.

### 2.4 Best-of-$n$ follows the top of the reward distribution

**Proposition 2 (extremal Goodhart under best-of-$n$).** Let a pool of behaviours have rewards $u_j$ and
values $v_j$, with distinct reward levels $w_1 < \dots < w_K$ and empirical CDF $F$. Best-of-$n$ with
replacement selects a behaviour at level $w_k$ with probability $F(w_k)^n - F(w_{k-1})^n$ (split evenly
among the ties), so
$$\mathbb E[v \mid \mathrm{Bo}(n)] \;\longrightarrow\; \mathbb E[v \mid u = \max_j u_j] \qquad (n \to \infty),$$
whatever the sign of ASD or of $\mathrm{Cov}_{\pi_0}(u, v)$.

*Example.* Honest arguments always earn $0.5$; dishonest ones earn $0.9$ with probability $0.1$ and $0$
otherwise. Then $\mathrm{ASD} = 0.5 - 0.09 = 0.41 > 0$, and for a base policy that argues honestly half of
the time $\mathrm{Cov}_{\pi_0}(u, v) > 0$: a small step of optimization helps. But
$$P(\mathrm{Bo}(n) \text{ is honest}) = 0.95^n - 0.45^n,$$
which rises from $0.5$ ($n=1$) to $0.77$ ($n=4$) and then falls to $0$: the optimum of the reward is always
a dishonest argument. This is *extremal Goodhart* (Manheim & Garrabrant 2018), and heavy-tailed reward
errors make it the generic case (Kwa et al. 2024). The library computes the selection exactly from sampled
pools (`analysis.optimization.BestOfN`; `"plugin"` is the with-replacement formula above, checked in the
tests), which is why optimization curves, not ASD alone, decide whether a mechanism is safe to optimize.

### 2.5 Ground truth by construction vs. by measurement

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
  propensities matter - and by the training algorithm (natural vs. vanilla policy gradient, section 8).
* **Zero-sum vs. general-sum.** Two-player zero-sum mechanisms (debate with zero-sum rewards) have a
  unique value (minimax theorem), which removes equilibrium-selection ambiguity *for the players'
  payoffs* - though not necessarily for $W$. General-sum mechanisms (teams, markets) typically have
  several equilibria, some good and some bad.
* **Does training reach equilibrium at all?** Simultaneous gradient play need not converge
  (cycling in zero-sum games is generic). PSRO (Lanctot et al. 2017) is a practical approximation:
  grow a population of strategies by best responses to the current meta-equilibrium and report
  NashConv (sum of regrets) as distance from equilibrium (`samplers.psro`).

**Selection over information sets.** Best-of-$n$ and tilting on a sampled game tree are selection
policies applied at each decision to the mover's expected payoff. Where a role cannot see an earlier
move - a simultaneous partner's, another role's private turn, a hidden draw - the payoff of each of its
candidates is an average over the histories it cannot tell apart, weighted by how likely the others'
behaviour makes each (its counterfactual reach), and one selection serves all of them: the agent-form
analogue of a quantal-response equilibrium (McKelvey & Palfrey 1998) with the selection policy as the
response function. Selecting separately behind each hidden move would grant the role the hidden
information - a clairvoyant response that no policy of the role can implement, overstating what
optimization achieves for it (a blind guesser of a coin would "win" every time). Without hidden moves
the solution is plain backward induction; with them it is the fixed point of fictitious play
(`analysis.optimization.evaluate_tree`, whose `gap` reports the distance from it).

**Optimizers that model each other.** When prompt search (an LLM optimizer) finds strategies for one
agent, its proposals depend on what it assumes about the others. PSRO makes that assumption explicit:
the optimizer is told the other roles' meta-strategy (strategies and probabilities) and asked for a
best response. A naive start followed by $k$ PSRO rounds is the analogue of level-$k$ reasoning.

### 3.1 The revelation principle, and why oversight needs indirect mechanisms

The revelation principle (Gibbard 1973; Myerson 1979) says that whatever some mechanism implements, a
*direct* one implements too: ask each agent for its private information and make truth-telling an
equilibrium. For oversight the direct mechanism is "ask the model for the answer and pay it for being
right". It fails, but not for strategic reasons: the principal cannot *compute* whether the answer is right,
so it cannot condition payments on correctness. The revelation principle assumes the designer can evaluate
the outcome function on reported types; in oversight the binding constraint is computational and
verificational, not informational. (It also fails in form when evidence is only partially verifiable - Green &
Laffont 1986, section 5.)

Indirect mechanisms get around this by turning the principal's job from *computing* into *checking*: debate
narrows a hard claim to one step the judge can verify (Irving, Christiano & Amodei 2018), doubly-efficient
debate keeps the honest side's cost polynomial (Brown-Cohen, Irving & Piliouras 2023), prover-verifier games
train the checker (Anil et al. 2021; Kirchner et al. 2024), and verified claims (section 5) hand the judge
trusted checks. The price is that truth-telling is no longer a property of a single report but of an
*equilibrium* of a game - everything in this section applies - and the reduction can fail: the
obfuscated-arguments problem (Barnes & Christiano 2020) is a dishonest debater hiding a flaw where no bounded
honest opponent can find it, so no single checkable step exists.

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
Ben-Porath, Dekel & Lipman 2019), agents can present evidence that rules out some lies. Information
economics gives a precise account of what verification buys an oversight protocol:

* **Cheap talk.** A sender whose preferred answer does not depend on the truth - an agent assigned a side,
  or trained to win - conveys nothing with unverifiable claims to a rational receiver: only babbling
  equilibria exist (Crawford & Sobel 1982, in the limit of maximal bias). A consultant arguing an assigned
  side with unverifiable claims cannot move a rational judge, so its ASD is 0; it does move a *credulous*
  judge, which is exactly what it can exploit.
* **Unraveling.** If evidence is verifiable, the receiver knows what could have been shown, and it is
  sceptical - it reads withheld evidence as bad news - then full disclosure is the unique equilibrium
  (Grossman 1981; Milgrom 1981). The argument fails if the sender may genuinely lack evidence (Dye 1985),
  and it fails if the receiver is naive: a selected sample of true facts, read as a random one, misleads.
* **Competing advocates.** With two opposed interested parties and verifiable information, even an
  *unsophisticated* decision-maker reaches the full-information decision (Milgrom & Roberts 1986; see also
  Shin 1998; Dewatripont & Tirole 1999): each side discloses what the other withholds, so competition
  substitutes for scepticism. This is the economic core of debate.
* **Limited verification.** When the receiver can check only part of the evidence, which parts to check is
  itself a design problem (Glazer & Rubinstein 2004, 2006); the revelation principle can fail under partial
  verifiability (Green & Laffont 1986), and costly verification becomes a budget-allocation problem
  (Ben-Porath, Dekel & Lipman 2014). A sender who can commit to how information is generated faces
  Bayesian persuasion instead (Kamenica & Gentzkow 2011).

**Proposition 3 (disclosure in HiddenBits).** $L$ odd, i.i.d. fair hidden bits; the question is whether
at least $m = (L+1)/2$ are 1. An advocate for an answer can make claims about single bits, of which its
first $k$ are checked by a truthful verifier. The true side has $s \ge m$ favourable bits, the false side
$L - s \le m - 1$. Write $o$ and $z$ for the numbers of verified ones and zeros shown. A *naive* judge treats
them as a random sample: $P(\text{yes}) = P(\mathrm{Bin}(L-o-z, \tfrac12) \ge m - o)$. A *sceptical* judge
knows they were selected: if an advocate reveals a uniformly random subset of its favourable bits, the
likelihood of the revealed set given $n$ ones is $\binom{n}{o}^{-1}$ on the $\binom{L-o}{n-o}$ configurations
containing it, and $\binom{L-o}{n-o}/\binom{n}{o} \propto \binom{L}{n}$, so only the bounds matter:
$$P(\text{yes}) = \frac{\sum_{n=\max(\ell, m)}^{h} \binom{L}{n}}{\sum_{n=\ell}^{h} \binom{L}{n}},
\qquad \ell = o,\; h = L - z,$$
with $h = o$ if the yes-advocate showed fewer than $k$ ones and $\ell = L - z$ if the no-advocate showed
fewer than $k$ zeros ("no more exist"). Then, with rewards the log score of the judge's probability on the
advocate's answer:

1. *Debate, any judge that treats verified bits as facts.* If $k \ge m$, the truthful debater establishes
   the answer with certainty whatever the other does: ASD is the log score's full range (clipped at
   $\varepsilon$: $\log\frac{1-\varepsilon}{\varepsilon}$).
2. *One advocate, sceptical judge.* If $k \ge m$ the judge learns the truth from either advocate: the liar's
   disclosure unravels, and ASD is again the full range.
3. *One advocate, naive judge.* Selective disclosure misleads: for $L = 15$, $k = 8$ the expected ASD is
   $0.037$.
4. *No verification.* A rational judge ($k = 0$, unverified claims ignored) learns nothing: ASD is exactly 0
   in every protocol.

*Proof.* (1) $m$ verified ones (or zeros) decide the question, and a false claim cannot be verified. (2) The
liar can show at most $L - s < m \le k$ favourable bits; showing fewer than $k$ pins its count to a
minority, while the truthful advocate shows $m$. (3) Direct computation over the prior
(`theory.disclosure.asd_table`). (4) The judge's posterior is its prior, $\tfrac12$ for odd $L$, whichever
side argues. $\square$

The domain `hidden_bits` implements this setting with scripted advocates and exact Bayesian judges
(naive, credulous with a `trust` in cheap talk, sceptical) that read only the verdicts they are shown;
the tests check their posteriors against the formulas episode by episode, and `demo_hiddenbits` traces
ASD against the budget (between the budget's extremes the scripted liars, who lie once their favourable
bits run out and are refuted within the budget, hand the judge extra bits). `demo_bon_budget` adds
optimization: best-of-$n$ on a lying debater under a 3-claim budget almost removes its lies from the
checked claims and moves them beyond the budget, where a credulous judge still believes them - a
checked-first-$k$ rule tells the optimizer where lies are safe (checking a random subset would not). With
every claim checked, selection removes most lies, yet a naive judge is still misled by the liar's
selected true bits: verification bounds what can be said, not what is left out.

Predictions the library is built to test:

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
  gap between audits of the two (`core.state.log_integrity`). The same holds for a claim's output: the
  output of code the claimant wrote is a message the claimant chose (`SELECT 'A is correct'`), so a claim
  is *verified* only when a trusted checker confirms what it states; one that merely ran is marked
  *executed* - cheap talk with a stamp on it, which a weak judge would otherwise read as proof.

Verification is therefore a first-class experimental axis (`core.verification`): *what* can be verified
(the verifiers), *whose* claims (`roles`), *how much* (a claim budget, or a cost budget with a cost per
verifier), *how reliably* (`noise`: an erring verifier shows a flipped verdict or a wrong output in exactly
the genuine format; ground truth keeps the correct one), *who sees the verdicts* (`show_to`) and how they
are displayed - and, in the judges, credulity towards unverified claims and scepticism about selective
disclosure. Noise draws are chance moves shared by every candidate of a decision, so best-of-$N$ never
selects the candidates whose checks happened to err.

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
`core.rewards` implements both the penalty and the inverse-probability-weighted forms. "Independently"
includes independence of the action being rewarded: the library draws audits from the item and seed, so
all candidates of a best-of-N decision share one draw - were each candidate audited separately, selection
would favour the unaudited ones (weighted reward 0, above every audited log score), optimizing audit luck. The same weighting
keeps *overseers* honest when only some of their calls are checked: a monitor paid the log score of its
suspicion only on audited episodes, when flagged episodes are always audited and unflagged ones rarely,
learns that not flagging is safe; paying $\mathbb 1[\text{audited}]\, s/P(\text{audited})$ restores a proper
scoring rule in expectation (`MonitoredWork(penalty="audit", audit_unflagged=q)`).

### 6.1 Monitors: moral hazard and manipulable signals

Monitoring protocols from AI control (trusted monitoring, defer-to-trusted, chain-of-thought and probe
monitors; Greenblatt et al. 2024) become mechanisms once their signals pay the agent: in `MonitoredWork`
the worker earns the grader's measured success minus a penalty driven by the monitor. Principal-agent
theory with hidden actions (moral hazard) says what to expect.

* **Informativeness** (Holmström 1979): the reward should condition on every signal that carries
  information about the hidden action beyond the others - output, monitor, chain of thought, probe
  readings (`monitor_sees`) - and on nothing else, which only adds noise.
* **Multitask moral hazard** (Holmström & Milgrom 1991): when some dimensions of the work are measured
  and others are not, strong incentives on the measured ones divert effort from the rest, so optimal
  incentives are *weaker*. This is reward hacking in economists' terms, and a principled rationale for KL
  regularization and early stopping when the reward measures only part of what we want.
* **Becker with limited liability.** An audit with probability $a$ and penalty $P$ deters a violation worth
  $G$ iff $aP \ge (1-a)G$ when a caught violator forfeits the gain, and iff $aP \ge G$ when it keeps it
  (Becker 1968; Mookherjee & Png 1989; `theory.monitoring.min_audit_rate`). "Audit rarely, punish hard"
  needs unbounded penalties; RL rewards are clipped and normalized - limited liability - so audits must be
  frequent, and with $P < G$ a kept gain is not deterred at all. Forfeiture matters as much as the penalty:
  a flagged worker in `MonitoredWork` earns $-\lambda$ (`forfeit=True`, the default), keeps its measured
  success minus $\lambda$ with `forfeit=False`, and loses its credit when its work is deferred to a trusted
  model (`defer=True`).
* **Strategic classification** (Hardt et al. 2016; Frankel & Kartik 2019): a signal that is informative
  about unoptimized behaviour loses information once agents are rewarded on it - they move the signal, not
  the behaviour. A monitor's suspicion is such a signal; optimizing against chain-of-thought monitors
  produces obfuscated reward hacking (Baker et al. 2025). A monitor's value must be measured under the
  optimization it will face.
* **Fail closed.** A monitor that gives no score must not count as a pass, or breaking the monitor (output
  that does not parse, an injection that derails it) becomes a strategy that beats every other form of
  misbehaviour. `MonitoredWork` treats a missing score or probe reading as maximal suspicion
  (`missing_score="flag"`) and counts them (`missing_scores`).

A minimal model shows the mechanics (`theory.monitoring`; the synthetic world `domains.monitoring`
implements it).

**Proposition 4 (penalizing alarms can buy obfuscation instead of good behaviour).** An actor chooses
honest work (measured success $s$), overt misbehaviour ($s + \Delta$) or covert misbehaviour
($s + \Delta - \kappa$; hiding costs $\kappa \ge 0$). A monitor raises an alarm with probabilities
$d_h \le d_c < d_o$, and the reward is measured success minus $\lambda \cdot \mathbb 1[\text{alarm}]$. Covert
beats overt misbehaviour iff $\lambda (d_o - d_c) > \kappa$, and beats honest work iff
$\Delta - \kappa > \lambda (d_c - d_h)$. So if hiding is cheap relative to the detection gap,
$\kappa/(d_o - d_c) < (\Delta - \kappa)/(d_c - d_h)$, then as $\lambda$ grows

* overt misbehaviour turns covert at $\lambda = \kappa/(d_o - d_c)$;
* misbehaviour stops only for $\lambda > (\Delta - \kappa)/(d_c - d_h)$ - never if covert misbehaviour is
  undetectable ($d_c = d_h$).

If hiding is expensive, covert misbehaviour is never optimal and overt misbehaviour stops at
$\lambda = \Delta/(d_o - d_h)$. Throughout the covert phase the alarm rate on misbehaving episodes is $d_c$,
not $d_o$: the monitor looks increasingly effective while the misbehaviour persists.

*Proof.* The expected rewards $s - \lambda d_h$, $s + \Delta - \lambda d_o$ and $s + \Delta - \kappa - \lambda d_c$
are lines in $\lambda$; the best response is their upper envelope, which starts with overt misbehaviour and
passes to ever flatter lines. Pairwise comparison gives the crossings $\lambda_{oc} = \kappa/(d_o - d_c)$,
$\lambda_{ch} = (\Delta - \kappa)/(d_c - d_h)$ and $\lambda_{oh} = \Delta/(d_o - d_h)$, and
$\lambda_{oh} = [(d_o - d_c)\lambda_{oc} + (d_c - d_h)\lambda_{ch}]/(d_o - d_h)$ is a weighted average of the
other two. So either $\lambda_{oc} < \lambda_{oh} < \lambda_{ch}$ and the covert line is on the envelope
between $\lambda_{oc}$ and $\lambda_{ch}$, or $\lambda_{ch} \le \lambda_{oh} \le \lambda_{oc}$ and honest work
overtakes overt misbehaviour before covert misbehaviour does. $\square$

With forfeiture a behaviour with measured success $m_a$ earns $m_a(1 - d_a) - \lambda d_a$ - still a line,
and `theory.monitoring.envelope` gives the switch points of any such menu; with $d_o = 1$, hiding then pays
even without a penalty.

The remedy is a detector whose accuracy does not respond to the agent's optimization. A simulated probe
(`integrations.probes.SimulatedProbe`) draws readings from ground truth with a fixed AUROC; thresholded,
it flags every kind of misbehaviour with the same probability $d_c = d_o = \mathrm{TPR}$ and honest work
with $d_h = \mathrm{FPR}$, so hiding never pays and misbehaviour stops at $\lambda = \Delta/(\mathrm{TPR} -
\mathrm{FPR})$. It stands in for a capability we might have (a robust deception probe); it reads ground
truth, so its uses are recorded (`Episode.gt_access`). Real detectors - LLM monitors, keyword rules, probes
fit on the agent's outputs (`integrations.probes.fit_linear_probe`) - can be optimized against. The gap
between the two, ground-truth misbehaviour against alarm rates under the same optimization pressure, is the
manipulation effect. `demos.demo_monitoring` measures it: with $s = 0.5$, $\Delta = 0.3$, $\kappa = 0.05$,
$d_o = 1$, $d_c = 0.15$, $d_h = 0.05$, a best-responding actor turns covert against a keyword monitor at
$\lambda \approx 0.059$ (theory $1/17$) and is deterred at $2.62$ (95% CI $2.44$-$2.82$; theory $2.5$),
while against a probe with AUROC 0.9 it never hides and is deterred at $0.467$ ($0.460$-$0.474$; theory
$0.472$).

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
shared reward. Supervisors and agents can collude, so mechanisms must be collusion-proof (Tirole 1986;
Laffont & Martimort 1997). Antitrust leniency programmes (Motta & Polo 2003; Spagnolo 2004; Harrington
2008) destabilize cartels by paying the first informant and forgiving the first confessor, turning
collusion into a race.

**Model** (`theory.whistleblower`). One member of a team commits a violation (e.g. a reward hack) that
adds $\Delta$ to every member's reward if it stands: $\Delta$ is each member's *stake* in it. Each of the
offender's $n$ teammates *witnesses* it independently with probability $o$ - in a swarm an agent sees
only part of the others' work, and does not know how many others saw it - and a witness reports it or
stays silent. A verified report reverts the violation and earns a bounty $s$, paid to each verified
reporter or shared among them. If no report is verified, an independent audit catches the violation
with probability $q$, and every member without a verified report pays $c$ (collective liability). Once
the violation is caught, every witness without a verified report pays $m$ (misprision) and the offender
pays $P$; $\beta$ is the offender's private benefit. In the library: `Team(observe_prob=o,
random_audit=q, self_reports=...)` and `Whistleblower(bounty=s, split=..., silence_penalty=c,
misprision=m, violation_penalty=P, leniency=...)`; `Team` records who witnessed whose work, and the
domain `synthetic_team` has `gain` ($\Delta$), `damage` (the ground-truth cost of a hack that stands) and
`opportunity` (who can hack).

Let every report be verified (reliability $\rho = 1$; the code handles $\rho < 1$, below) and let $K$ be
the number of *other* witnesses who report. Reporting gains a witness
$$A(K) = \begin{cases} s - T, & K = 0,\\ m + s_K, & K \ge 1,\end{cases} \qquad T = (1-q)\,\Delta - (c+m)\,q,$$
with $s_K = s/(1+K)$ for a shared bounty and $s_K = s$ for one paid to each reporter. $T$ is the stake net
of what an audit would cost the silent: exposing a violation that nobody else would have exposed costs
$\Delta$ unless the audit would have found it anyway. If every witness reports with probability $p$, a
witness faces $K \sim \mathrm{Bin}(n-1, op)$. Write $P_k = \Pr[\mathrm{Bin}(n-1, o) = k]$ and
$\bar A(p) = \mathbb E\,A(K)$ (`report_gain`).

**Proposition 5 (reporting under partial observation).** Let the bounty be shared (for one paid to each
reporter, replace $s/(1+k)$ by $s$), and let $s > 0$ or $m > 0$.

1. If $s > T$, reporting is a best response to whatever the others do, and everyone reporting is the
   unique equilibrium (`dominance_bounty`).
2. If $s < T$, universal silence is a strict equilibrium. Everyone reporting is an equilibrium iff
   $s \ge b_R$ (`report_equilibrium_bounty`), where
   $$b_R = \frac{P_0\,T - (1 - P_0)\,m}{P_0 + \sum_{k \ge 1} P_k/(1+k)}.$$
   - For $b_R < s < T$ both are strict equilibria, separated by exactly one mixed equilibrium. Whether
     snitching pays depends entirely on whether a witness expects others to snitch.
   - For $s < b_R$ silence is the *only* equilibrium, although the bounty is positive. A witness may be
     the only one ($P_0 = (1-o)^{n-1} > 0$), and the bounty does not pay for the risk of reporting
     alone. In a team of three ($n = 2$ witnesses) with $o = 0.8$ and $q = m = 0$, $b_R = \Delta/3$.
3. With full observation ($o = 1$) $P_0 = 0$, so everyone reporting is an equilibrium at any bounty
   (weakly at $s = m = 0$). With a bounty paid to each reporter and $q = c = m = 0$, the mixed equilibrium
   is $p^* = 1 - (s/\Delta)^{1/(n-1)}$, and training that starts with report propensity below $p^*$ ends
   in universal silence. The silent basin *shrinks with team size* ($p^* \to 0$ as $n \to \infty$): in a
   large team someone else will probably report, so you should too. For two witnesses the game is a
   $2 \times 2$ coordination game in which reporting is risk dominant (Harsanyi & Selten 1988) iff
   $\tfrac32 s + m > T$ (`risk_dominant`). For $n \ge 3$ there is no comparable selection result:
   $A(K)$ *falls* in $K \ge 1$ when bounties are shared, so the game lacks the strategic complementarity
   that global-games selection needs (Carlsson & van Damme 1993; Morris & Shin 2003). Which equilibrium
   training reaches is then a question of basins (Proposition 6).
4. **Common reward** ($s = c = m = 0$). Silence weakly dominates reporting, strictly when nobody else
   reports ($A(0) = -(1-q)\Delta$): training on the shared objective pushes every agent to stay silent
   about a hack that helps the team. Anticipating silence, a member with the opportunity violates
   whenever $\beta + (1-q)\Delta - qP > 0$; under a purely shared reward ($P = 0$) that is always.
   Every member is a stakeholder in undetected misconduct.

*Proof.* (1) $A(0) = s - T > 0$, and $A(k) = m + s/(1+k) > 0$ for $k \ge 1$. (2) $\bar A(0) = A(0) < 0$,
and $\bar A(1) = \sum_k P_k A(k)$ is affine and increasing in $s$ with root $b_R$. For the mixed
equilibrium, put $r = op$, $N = n - 1$ and $t = r/(1-r)$. Then
$\bar A = (1-r)^N \sum_{k=0}^{N} \binom Nk A(k)\,t^k$, whose coefficients change sign exactly once
($A(0) < 0 < A(k)$ for $k \ge 1$). By Descartes' rule of signs $\bar A$ has at most one zero with $t > 0$,
and it has one in $0 < p < 1$ iff $\bar A(1) > 0$. (The expected advantage need not be monotone in $p$.)
(3) $P_0 = 0$ gives $\bar A(1) = \sum_{k \ge 1} P_k A(k) \ge 0$; with each-reporter bounties
$\bar A(p) = s - \Delta (1-p)^{n-1}$. In the $2\times2$ game a deviation from (report, report) costs
$A(1) = m + s/2$ and one from (silent, silent) costs $-A(0) = T - s$; the equilibrium with the larger
product of the two players' losses risk-dominates. (4) Substitute; a violator's gain when nobody
reports is $\beta + (1-q)\Delta - qP$. $\square$

So a bounty must exceed each agent's *stake* in the undetected violation, not merely be positive, to make
reporting safe whatever the others do. Below the stake it creates a coordination game only if a witness
is unlikely to be alone; otherwise it achieves nothing. The stake is the full team score when everyone is
paid the team score. Audits, collective liability and misprision lower both thresholds (collective
liability alone makes reporting dominant once $s + (\Delta + c)q > \Delta$). Only a *verified* report
counts: a refuted or unverifiable accusation does not spare its author from $c$ or $m$ (and pays any
false-report penalty) - otherwise blaming a teammate at random, or for the violator blaming anyone,
would escape the liability for free. Paying only the first reporter and forgiving a violator who
reports itself (`leniency`) turns a cover-up into a race.

With unreliable verification ($\rho < 1$, $\bar\rho = 1 - \rho$) and a false-report penalty $f$ for a
refuted report, $A(k) = \rho\bar\rho^{k}\,[cq - (\Delta + m)(1-q)] + \rho m - \bar\rho f + B(k)$, where
$B(k) = s\rho$ (each) or $s(1 - \bar\rho^{k+1})/(k+1)$ (shared) is the expected bounty. With each-reporter
bounties $A$ is monotone in $k$, so there is still at most one mixed equilibrium; with shared bounties and
$f > 0$ it can change sign twice, and `symmetric_equilibria` and `regime` find the equilibria numerically.

**Nobody violating can be a Nash equilibrium without being subgame perfect.** Proposition 5 is about the
*reporting subgame*, played after a violation. In the whole game, where each member fixes its strategy
(violate if you can? report what you see?) in advance (`team_game`), "nobody violates and everyone would
report" is a Nash equilibrium whenever the threat deters, $G(1) < 0$ below - even for $s < b_R$, since
on the equilibrium path no report is ever made. It is not subgame perfect (Selten 1975): a witness who
actually sees a violation prefers silence. With the synthetic team's numbers ($\Delta = P = 0.3$,
$o = 0.8$, two witnesses, shared bounty) and $s = 0.05 < b_R = 0.1$, the profile is a Nash equilibrium of `team_game`
while everyone reporting is not one of the reporting subgame (tested). Training that only sees episodes
where nobody violates never tests the threat, and nothing in the training signal maintains it, so such an
honest outcome is fragile.

**Proposition 6 (the whole game's learning dynamics are two-dimensional).** Let members be symmetric;
each holds the opportunity to violate with probability $\pi$ ($\pi = 1/(n+1)$ when one random member of
the $n + 1$ does) and then violates with probability $x$; a witness reports with probability $p$. A
violation is caught with probability $d(p) = 1 - (1-q)(1-op)^n$, so violating gains the offender
$$G(p) = \beta + \Delta\,(1 - d(p)) - P\,d(p) - c\,q\,(1-op)^n$$
(`offender_gain`). Natural policy gradient (Kakade 2001) on a softmax over the four strategies
(exponential weights, whose mean-field limit is the replicator dynamics) keeps the two decisions
independent, and the marginals follow
$$\operatorname{logit} x \mathrel{+}= \eta\,\pi\,G(p), \qquad
  \operatorname{logit} p \mathrel{+}= \eta\,(1-\pi)\,x\,o\,\bar A(p).$$

*Proof.* A member's payoff is additively separable in its two decisions: the first matters only when it
holds the opportunity, the second only when a teammate violated and it saw it. So the update multiplies a
strategy's weight by $e^{\eta(u_V(v) + u_R(r))} = e^{\eta u_V(v)}\,e^{\eta u_R(r)}$, and a product
distribution stays a product. Summing over the other decision gives the marginal updates, with
$u_V(\text{violate}) - u_V(\text{honest}) = \pi\,G(p)$ and
$u_R(\text{report}) - u_R(\text{silent}) = (1-\pi)\,x\,o\,\bar A(p)$. $\square$

`games.learning.policy_gradient(team_game(n), natural=True)` with shared logits reproduces
`whistleblower.mean_field` exactly (tested); `basin_of_deterrence` is the share of initial report
propensities from which misconduct ends deterred.

**Equilibrium selection depends on the training algorithm.** Vanilla policy gradient (the expected
REINFORCE update; Williams 1992) has the same rest points, but moves a strategy's logit by $\eta\,\pi_a(u_a - \bar u)$
instead of $\eta\,(u_a - \bar u)$: a rarely played strategy - reporting, when nobody reports - learns
slowly, and the path of the whole game depends on the race between its two decisions. With the numbers
above and $s = 0.45$ (above the stake), starting from $x = 0.5$ and $p = 0.1$ with step size 2, natural
gradient deters misconduct ($x < 0.5$) after 29 steps. Vanilla gradient first entrenches it ($x$ peaks
at 0.89) and brings $x$ below 0.5 only after 227 steps (tested). In the coordination region the basins
differ: at $s = 0.15$, from $x = 0.5$ and $p = 0.6$, natural gradient deters misconduct while vanilla
gradient ends in "violate and stay silent" (tested). `games.learning.StrategyGradient` trains on sampled
episodes instead, with unbiased estimates of either update, and `demo_swarm` averages it over runs.
Robustness to training is a property of the mechanism *and* the training algorithm.

The simulated team game in `domains.synthetic_team` reproduces these payoffs: exactly with full
observation, and in expectation over who sees what with `Team(observe_prob=o)`. The same code runs with
LLM agents (the code domain provides a team task whose grader runs only visible tests, so a teammate can
hard-code them).

## 9. Robustness to training

* **Samplers form an adversary hierarchy**: instructed arms < natural behaviour under best-of-N <
  prompt search (directive-constrained, reaching behaviour far from the base policy; parameter search for
  programmatic agents, `samplers.param_search`) < PSRO < RL. For continuous ground truth the ground truth a
  perfect optimizer of the reward gives up within a searched class is its *regret* (`metrics.gt_regret`).
  "Robust IC at level $k$" means IC against adversary class $k$; the library measures each level.
* **Validity of cheap proxies**: does prompted ASD predict what RL-trained policies do? Measurable by
  correlating prompted ASD across mechanisms with post-training ground-truth values.
* **Fixtures are attack surfaces**: optimizing against a fixed judge finds its exploits (length,
  confidence, jargon, a speaking slot, an option label); optimization curves expose them, and
  `analysis.diagnostics` measures each at fixed ground truth before anyone optimizes.
* **Exploration**: RL only discovers what it explores; deceptive strategies far from the base policy may
  never be sampled (the limitation of best-of-N), and models may under-explore deliberately.
* **Chain-of-thought as oversight channel**: penalizing suspicious reasoning teaches concealment; a
  monitor's value must be measured after optimization against it (`mechanisms.MonitoredWork` with
  `monitor_sees=("reasoning",)`; Proposition 4 in section 6.1).

### 9.1 Collusion: log-score debate is not zero-sum

Nash equilibria rule out *unilateral* deviations; agents trained by one process can also deviate *jointly*.
The ASD-style debate rewards $(\log p_A, \log p_B)$ with $p_A + p_B = 1$ are strictly competitive over
outcomes - whatever helps one debater hurts the other - but not zero-sum, and their mixed extension is not
strictly competitive either: games that are strictly competitive in mixed strategies are exactly the affine
transformations of zero-sum games (Adler, Daskalakis & Papadimitriou 2009). The sum is what self-play
optimizes when one policy occupies both seats, and it is largest when the judge stays undecided:
$2\log\tfrac12 \approx -1.39$, against $\log 0.99 + \log 0.01 \approx -4.62$ for a vigorous debate the honest side
wins. So a shared policy gains by keeping debates soft - collusion that no unilateral-deviation analysis
sees. Antisymmetric rewards remove the incentive, since the sum is constant (`Debate(zero_sum=True)`,
`rewards.ZeroSum`); `NormalFormGame.coalition_deviations` measures the largest joint gain in an empirical
game. Swarms are the extreme case (section 8): a shared objective makes every member a stakeholder in
covering up the others.

### 9.2 Learning dynamics select equilibria

Multiple equilibria are the norm, and training is the selection device: fictitious play and no-regret
learning (Fudenberg & Levine 1998), evolutionary dynamics and their basins (Sandholm 2010), and
*stochastic stability* - learning with small persistent noise (exploration) spends almost all its time at
the risk-dominant equilibrium of a $2\times2$ coordination game (Kandori, Mailath & Rob 1993; Young 1993). That
links section 8's risk-dominance and global-games predictions to training with exploration, and it is one
more reason the training algorithm is part of the mechanism's environment (Proposition 6).

When several roles are optimized, one optimizer's best move depends on how it models the others. The
standard solution concepts are PSRO meta-solvers (Lanctot et al. 2017; `samplers.psro`), each a model of
the opponents:

| Meta-solver | Best response to | Classical counterpart |
|---|---|---|
| `last` | the opponents' newest strategy | iterated best response: level-$k$ reasoning (Stahl & Wilson 1995; Nagel 1995; Camerer, Ho & Chong 2004) |
| `uniform` | the uniform mixture of all strategies so far | fictitious play (Brown 1951) |
| `replicator` | the rest point replicator dynamics reach from uniform | evolutionary selection |
| `nash` | a Nash equilibrium of the restricted game | double oracle (McMahan, Gordon & Blum 2003) |

The oracle can be an LLM prompt optimizer told the opponents' meta-strategy, a fixed pool or a parameter
search; each PSRO iteration records the meta-equilibrium's ground truth, so a trace shows whether the
strategic landscape pulls toward honesty or deception as the optimizers adapt to each other.

### 9.3 How oversight differs from classical mechanism design

Section 1 named two differences; in full:

1. **Utility is the reward.** The designer writes the agents' objectives but not their policy class or
   optimizer, so IC must hold over *reachable* policies, not all policies.
2. **The principal is computationally bounded.** It cannot compute the social choice function from
   reports, so indirect, verification-based mechanisms are necessary (section 3.1).
3. **Optimization is bounded, local and path-dependent.** Agents follow gradients from an initial policy;
   equilibrium selection is dynamic (section 9.2), and the training algorithm is part of the environment.
4. **Agents are copies.** Types and strategies are correlated, collusion is free (section 9.1), and
   peer-based mechanisms lose their independence assumptions (section 7).
5. **Rewards are bounded and normalized** (limited liability, section 6.1), and training statistics -
   baselines, normalization across a batch - couple agents' incentives in ways payments do not.
6. **We care about what agents become.** The principal's objective includes each role's behaviour (honesty,
   not tampering), not only the decision in the current episode.
7. **Participation is not voluntary, but abstention is a strategy.** Individual rationality constraints
   disappear; refusing and hedging remain and must be rewarded sensibly.
8. **Counterfactuals are cheap.** The same agent can be rerun on the same item under other mechanisms,
   prompts and seeds, and recorded episodes re-scored under other reward rules (`rescore`). Mechanism
   evaluation becomes an empirical science of *measured* incentive compatibility, with a ground-truth oracle
   held back - the premise of this library.

## 10. What matters besides incentive compatibility

Incentive compatibility says optimizing the mechanism's reward moves agents toward the truth. A
useful mechanism also needs the following ([beyond_ic.md](beyond_ic.md) turns each into a measurement recipe
with the library's functions):

1. **Outcome quality at equilibrium** - IC with useless outcomes is pointless; control-style accuracy
   still matters, evaluated at the equilibrium training reaches (`metrics.judge_accuracy`, game outcomes).
2. **Robustness to collusion** - coalitional deviations and shared-weights agents
   (`NormalFormGame.coalition_deviations`, `directive="collude"`, team games).
3. **Reachability and selection** - does training reach the good equilibrium from realistic
   initializations (basins, PSRO NashConv)?
4. **Signal quality** - reward variance, sparsity and credit assignment determine sample efficiency
   (label efficiency, standard errors, `diagnostics.reward_snr`).
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
    uncertainty (use proper scoring rules; check calibration, `diagnostics.calibration`).

## 11. Open theoretical questions

* When does debate with a bounded, fallible arbiter and limited verification implement the truth
  (Moore-Repullo with imperfect arbitration)? Full implementation: when is *every* reachable equilibrium
  of a debate among bounded agents honest, obfuscated arguments included?
* **Reachability.** How well do prompt and parameter search proxy what weight-level RL reaches, and how can
  a searched frontier be validated against actual training?
* **Obfuscation-robust monitoring.** Which monitoring signals keep their informativeness under optimization
  against them (Proposition 4)?
* **Selection-aware judges.** Unraveling needs sceptical receivers (Proposition 3); can judges be trained to
  infer from what was *not* shown?
* **Swarm design at scale.** Optimal combinations of bounties, audits, leniency and misprision under
  realistic observation structures, when members share weights and can coordinate implicitly.
* **Delayed ground truth in training.** Credit assignment and variance when the proper reward arrives long
  after the behaviour; releases as a public ground-truth channel.
* Sample complexity of training to an honest equilibrium given reward noise - label efficiency as a
  learning-theoretic quantity.
* Collusion-proofness for agents that share weights or training data.
* Robust mechanism design (Bergemann & Morris 2005) against unknown judge biases.
* Which mechanisms keep $\mathrm{Cov}_{\pi_\beta}(u, v) \ge 0$ for all $\beta$ - i.e. are Goodhart-free
  along the whole KL-regularized optimization path?

## References

Adler, Daskalakis & Papadimitriou (2009), A note on strictly competitive games, *WINE*.
Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (2023), Prediction-powered inference, *Science*.
Anil, Zhang, Wu & Grosse (2021), Learning to give checkable answers with prover-verifier games.
Azar & Micali (2012), Rational proofs, *STOC*.
Baker et al. (2025), Monitoring reasoning models for misbehavior and the risks of promoting obfuscation.
Barnes & Christiano (2020), Debate update: obfuscated arguments problem, *AI Alignment Forum*.
Becker (1968), Crime and punishment: an economic approach, *JPE*.
Beirami et al. (2024), Theoretical guarantees on the best-of-n alignment policy.
Ben-Porath, Dekel & Lipman (2014), Optimal allocation with costly verification, *AER*.
Ben-Porath, Dekel & Lipman (2019), Mechanisms with evidence: commitment and robustness, *Econometrica*.
Bergemann & Morris (2005), Robust mechanism design, *Econometrica*.
Brown (1951), Iterative solution of games by fictitious play, in *Activity Analysis of Production and Allocation*.
Brown-Cohen, Irving & Piliouras (2023), Scalable AI safety via doubly-efficient debate.
Brown-Cohen et al. (2025), Avoiding obfuscation with prover-estimator debate.
Bull & Watson (2007), Hard evidence and mechanism design, *GEB*.
Camerer, Ho & Chong (2004), A cognitive hierarchy model of games, *QJE*.
Carlsson & van Damme (1993), Global games and equilibrium selection, *Econometrica*.
Chen, McCauley & Singh (2016), Rational proofs with multiple provers, *ITCS*.
Chen, Dimitrov, Sami, Reeves, Pennock, Hanson, Fortnow & Gonen (2010), Gaming prediction markets, *Algorithmica*.
Crawford & Sobel (1982), Strategic information transmission, *Econometrica*.
Dasgupta & Ghosh (2013), Crowdsourced judgement elicitation with endogenous proficiency, *WWW*.
Dewatripont & Tirole (1999), Advocates, *JPE*.
Dye (1985), Disclosure of nonproprietary information, *J. Accounting Research*.
Frankel & Kartik (2019), Muddled information, *JPE*.
Fudenberg & Levine (1998), *The Theory of Learning in Games*.
Gao, Schulman & Hilton (2023), Scaling laws for reward model overoptimization, *ICML*.
Gibbard (1973), Manipulation of voting schemes: a general result, *Econometrica*.
Glazer & Rubinstein (2004), On optimal rules of persuasion, *Econometrica*; (2006) A study in the pragmatics of persuasion, *Theoretical Economics*.
Green & Laffont (1986), Partially verifiable information and mechanism design, *RES*.
Greenblatt, Shlegeris, Sachan & Roger (2024), AI control: improving safety despite intentional subversion, *ICML*.
Grossman (1981), The informational role of warranties and private disclosure, *JLE*.
Hanson (2003), Combinatorial information market design; (2007) Logarithmic market scoring rules.
Harrington (2008), Optimal corporate leniency programs, *JIE*.
Hardt, Megiddo, Papadimitriou & Wootters (2016), Strategic classification, *ITCS*.
Harsanyi & Selten (1988), *A General Theory of Equilibrium Selection in Games*.
Hart, Kremer & Perry (2017), Evidence games: truth and commitment, *AER*.
Holmström (1979), Moral hazard and observability, *Bell J. Econ.*; (1982) Moral hazard in teams, *Bell J. Econ.*
Holmström & Milgrom (1991), Multitask principal-agent analyses, *JLEO*.
Hubinger (2020), AI safety via market making, *AI Alignment Forum*.
Irving, Christiano & Amodei (2018), AI safety via debate.
Kakade (2001), A natural policy gradient, *NeurIPS*.
Kamenica & Gentzkow (2011), Bayesian persuasion, *AER*.
Kandori, Mailath & Rob (1993), Learning, mutation, and long run equilibria in games, *Econometrica*.
Kirchner et al. (2024), Prover-verifier games improve legibility of LLM outputs.
Kwa, Thomas & Garriga-Alonso (2024), Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification.
Laffont & Martimort (1997), Collusion under asymmetric information, *Econometrica*.
Lanctot et al. (2017), A unified game-theoretic approach to multiagent reinforcement learning, *NeurIPS*.
Manheim & Garrabrant (2018), Categorizing variants of Goodhart's law.
Maskin (1999), Nash equilibrium and welfare optimality, *RES*.
McKelvey & Palfrey (1998), Quantal response equilibria for extensive form games, *Experimental Economics*.
McMahan, Gordon & Blum (2003), Planning in the presence of cost functions controlled by an adversary, *ICML*.
Milgrom & Roberts (1986), Relying on the information of interested parties, *RAND J. Econ.*
Milgrom (1981), Good news and bad news: representation theorems and applications, *Bell J. Econ.*
Miller, Resnick & Zeckhauser (2005), Eliciting informative feedback: the peer-prediction method, *Management Science*.
Mookherjee & Png (1989), Optimal auditing, insurance, and redistribution, *QJE*.
Moore & Repullo (1988), Subgame perfect implementation, *Econometrica*.
Morris & Shin (2003), Global games: theory and applications, in *Advances in Economics and Econometrics*.
Motta & Polo (2003), Leniency programs and cartel prosecution, *IJIO*.
Myerson (1979), Incentive compatibility and the bargaining problem, *Econometrica*.
Nagel (1995), Unraveling in guessing games: an experimental study, *AER*.
Paleka, Pallavi Sudhir et al. (2025), Consistency checks for language model forecasters, *ICLR*.
Pallavi Sudhir, Kaunismaa & Panickssery (2025), A benchmark for scalable oversight mechanisms (ASD), arXiv:2504.03731.
Prelec (2004), A Bayesian truth serum for subjective data, *Science*.
Prelec, Seung & McCoy (2017), A solution to the single-question crowd wisdom problem, *Nature*.
Sandholm (2010), *Population Games and Evolutionary Dynamics*.
Selten (1975), Reexamination of the perfectness concept for equilibrium points in extensive games, *IJGT*.
Shin (1998), Adversarial and inquisitorial procedures in arbitration, *RAND J. Econ.*
Shnayder, Agarwal, Frongillo & Parkes (2016), Informed truthfulness in multi-task peer prediction, *EC*.
Spagnolo (2004), Divide et impera: optimal leniency programmes.
Stahl & Wilson (1995), On players' models of other players: theory and experimental evidence, *GEB*.
Tirole (1986), Hierarchies and bureaucracies: on the role of collusion in organizations, *JLEO*.
Townsend (1979), Optimal contracts and competitive markets with costly state verification, *JET*.
Wellman (2006), Methods for empirical game-theoretic analysis, *AAAI*.
Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, *Machine Learning*.
Young (1993), The evolution of conventions, *Econometrica*.
