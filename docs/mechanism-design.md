# Scalable oversight as empirical mechanism design

The framework below is the research model used by this package. The bounds and swarm calculation are elementary derivations under stated assumptions, not claims established by the demo.

## The mechanism induces a game

Let $x$ be a task, $z$ evaluator-only state, $\theta_i$ an agent's private information, and $\pi_i$ its policy. A mechanism $M$ specifies observations, communication, actions, verification access, a terminal outcome $y$, and utilities $R_i(\tau)$ on trajectories $\tau$. A separate evaluator defines desirable behavior $G_i(\tau,z)$ and collective value $W(y,z)$.

The experimental object is

$$
\tau \sim P_M(\cdot\mid x,\theta,\pi_1,\ldots,\pi_k),\qquad
(R_1,\ldots,R_k;G_1,\ldots,G_k;W).
$$

A fixture's policy is fixed; a trainable role optimizes its expected $R_i$. The fixture/trainable distinction is independent of model capability and trust. Costs, risk aversion, shared weights and outside objectives change effective utility; modelling agents as maximizing a scalar mechanism reward is an experimental assumption to test.

Classical mechanism design studies rules that elicit private information and induce desired outcomes under strategic behavior. Dominant-strategy compatibility quantifies over other agents' strategies; Bayesian compatibility averages over specified beliefs. Revelation arguments depend on their information and rationality assumptions and do not automatically make computationally bounded, tool-using LLM workflows truthful. [Mechanism design background](https://www.nobelprize.org/uploads/2018/06/advanced-economicsciences2007-2.pdf).

Here, truthful *reporting*, costly truthful *investigation*, competent *execution*, and beneficial *outcomes* may differ. An agent can sincerely report a poor belief or produce correct code while concealing sabotage. Define the behavioral dimensions before choosing a benchmark label.

## Four different incentive questions

Fix a distribution over tasks and peers, and let $U_i(\pi_i,\pi_{-i})=\mathbb{E}[R_i]$.

1. **A paired incentive margin** compares specified good/bad interventions: $\Delta_i=U_i(\pi_i^+,\pi_{-i})-U_i(\pi_i^-,\pi_{-i})$. ASD measures this kind of contrast. Positive margins for a finite set do not rule out profitable untested deception.
2. **Reward regret** asks whether a policy is a best response:
   $$\operatorname{Reg}_i=\sup_{\pi_i'}U_i(\pi_i',\pi_{-i})-U_i(\pi_i,\pi_{-i}).$$
   Small regret says optimization has approximately succeeded; it says nothing by itself about desirability.
3. **Alignment of reward optima** asks whether high-reward policies have high $G_i$, including the *worst* desirable value among approximate reward maximizers. Define
   $$A_i(\epsilon)=\inf_{\pi_i:U_i(\pi_i)\ge\sup_{\pi_i'}U_i(\pi_i')-\epsilon}\mathbb{E}[G_i(\pi_i)].$$
   The infimum matters because a mechanism may have both honest and deceptive optima.
4. **Learning dynamics and equilibrium selection** ask which joint policy is reached, from which initial population and beliefs, under a specified update rule. An aligned equilibrium can coexist with unaligned equilibria or unstable dynamics.

`FiniteGame.regrets` enumerates unilateral deviations on the supplied finite strategy support. It is a lower bound on exploitability in a richer class, not a certificate of global incentive compatibility. Sampling more reward-maximizing strategies and more behaviorally diverse strategies answers different questions; combine both.

## A sufficient alignment bound

For fixed peers, suppose throughout the *entire* candidate class

$$|U_i(\pi)-a_i V_i(\pi)-b_i|\le\delta_i,\qquad
V_i(\pi)=\mathbb{E}[G_i(\pi)],\quad a_i>0.$$

If $\hat\pi$ is $\epsilon$-optimal for reward, then

$$\sup_\pi V_i(\pi)-V_i(\hat\pi)\le(\epsilon+2\delta_i)/a_i.$$

To see this, for any $\pi$, subtract the two approximation inequalities to get
$a_i[V_i(\pi)-V_i(\hat\pi)]\le U_i(\pi)-U_i(\hat\pi)+2\delta_i\le\epsilon+2\delta_i$, then take the supremum. The statement does not require an attained optimum.

This is a useful target, not an empirical guarantee: average correlation does not imply a uniform approximation bound. Optimizing agents can search precisely where the approximation fails. If the condition holds only against one peer profile, simultaneous updates can invalidate it. A stronger claim must quantify over the relevant peers, coalitions, task shifts and verifier errors.

## Local optimization and Goodhart effects

For a fixed finite policy distribution $p_0(a)$ and fixed peers, consider exponential reward tilting:

$$p_\beta(a)=\frac{p_0(a)e^{\beta R(a)}}{\sum_b p_0(b)e^{\beta R(b)}}.$$

Differentiating the expectation gives

$$\frac{d}{d\beta}\mathbb{E}_{p_\beta}[G]=\operatorname{Cov}_{p_\beta}(G,R),\qquad
\frac{d}{d\beta}\mathbb{E}_{p_\beta}[R]=\operatorname{Var}_{p_\beta}(R)\ge0.$$

Reward can therefore rise while true quality falls. A positive initial covariance need not stay positive at higher pressure. `optimization_response` estimates the zero-pressure covariance within tasks; plots and targeted prompt search probe beyond that local point. This derivation describes a reweighted finite distribution, not a theorem about neural-network gradient dynamics.

For empirical BoN, if $F(r)$ is cumulative base mass up to reward $r$, a tied reward group's probability is $F(r)^n-F(r^-)^n$. Allocate group probability proportionally to its base masses. For nested games, first compute each child's continuation utilities; the current player tilts using its *own* continuation reward. This assumes observable decision histories. Simultaneous play needs joint best-response analysis rather than pretend sequential information.

## Reporting incentives can create a coordination problem

Consider an already discovered, verifiable violation. Every worker receives common payoff $L$ if nobody reports and $0$ if anyone reports. A valid reporter receives individual bonus $b$ and pays reporting cost $c$. Let $q$ be a worker's belief that at least one *other* worker reports. Then

$$U(\text{report})=b-c,\qquad U(\text{silent})=(1-q)L,$$

so reporting is profitable exactly when

$$b-c>(1-q)L.$$

When $0<b-c<L$, both universal silence and universal reporting are pure equilibria. The latter is not necessarily coalition-stable: if all workers jointly move to silence, each replaces $b-c$ with $L$. `reporting_game(0.2)` in the demo uses $L=1,c=0$ and recovers both equilibria and this coalition gain. Increasing $b-c$ above $L$ makes reporting strictly dominant in *this model*, at a corresponding budget cost. Negative/retaliatory transfers or a bounty shared across reporters change the calculation.

Real experiments should vary report visibility, first-reporter versus all-reporter payment, audit availability, discovery effort, false reports, coalition size, shared training and whistleblower protection. Do not assume the target violation is perfectly known. If a false report passes an audit with probability $\alpha$ and incurs penalty $f$ when rejected, its expected incremental payment before task effects is

$$\alpha b-(1-\alpha)f-c.$$

A negative value discourages that particular false-report strategy under the model assumptions. It does not address evidence fabrication, verifier corruption, sybil reports, retaliation or detection avoidance. `Swarm` rejects duplicate reporter/target pairs, supports shared bounties and audits, and allows the researcher to choose a different workflow/payoff rule for those extensions.

## Trainable judges and verification limits

A judge's reward should encourage whatever role it performs: calibrated beliefs, accurate audits, useful information requests, or faithful application of a rule. Agreement with rewarded agents is not generally sufficient; mutually reinforcing false beliefs can form equilibria. Delayed verified outcomes can supply proper scores for judge training on a resolved subset, with the selection of that subset part of the mechanism.

Proper scoring rules make truthful probabilistic belief reports optimal in expectation when the scored outcome is generated from the forecaster's belief distribution and the utility is the expected score. This does not itself reward costly information acquisition or guarantee model calibration under distribution shift. [Gneiting and Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

There is also a simple identification limit. If two behaviors induce the same distribution over **all signals available to the reward mechanism**, any reward measurable from those signals has the same distribution and expectation for both behaviors. Such a mechanism cannot strictly prefer one behavior. This follows directly by applying the reward function to identically distributed observable signals. The assumption can fail when tools, random audits, private commitments, outside information or future outcomes reveal a distinction; that is precisely why these are experimental axes rather than decorative features.

## Claims the package deliberately does not make

No finite dataset establishes dominant-strategy honesty for arbitrary future agents. The normal-form utilities describe independent mixed profiles; correlated-equilibrium constraints, side-payment negotiation, partial-observation equilibrium solvers and full neural weight optimization are not implemented. A protocol can nevertheless be run with private information or shared-model agents and its empirical deviations measured. A strong research result should state its task distribution, policy/search class, peer model, utility, verification assumptions and learning dynamics alongside the numerical results.
