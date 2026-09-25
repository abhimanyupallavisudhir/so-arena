# How the design answers the brief

## 1. Arbitrary mechanisms with any reward computation

A mechanism is a **game form plus a reward rule** (`core.mechanism.Mechanism`). The game form is
ordinary async Python calling `g.act(role, ...)`; roles are trainable agents or fixtures; each role's
information (affordances, visibility, others' chain of thought), capabilities (tools) and verifiable
claims are configured per role. Reward rules are separate objects, so one protocol can be paired with
many reward rules, and rules can be re-applied to logged episodes.

**Bespoke or general?** General mechanisms plus domain adapters, as in ControlArena's settings ×
micro-protocols. Mechanisms render items generically and delegate domain logic to *verifiers* (trusted
checks) and *tools* (capability gaps); domain-specific mechanisms remain possible by subclassing.

**What from ControlArena makes sense with rewards?** Its micro-protocols become mechanisms once the
untrusted agent is paid something: task success minus a suspicion penalty, or task success unless
flagged/caught by an audit (`integrations.controlarena`). These are trainable, with the known caveat
that optimizing against a monitor teaches evasion - which the library measures rather than assumes
(ASD after optimization; CoT-visible monitors in `MonitoredWork`). Existing ControlArena logs yield
ASD for free; a common finding it makes visible is that the plain task reward gives *no* training
incentive against attacking (attacks also complete the main task). Monitors on actions, chain of
thought, tool calls and probe readings (`monitor_sees=`, `action.metadata["probe_scores"]` from a
white-box backend) are built in; any SO-arena mechanism can also run *as* a ControlArena monitor.

## 2. Scoring behaviours and sampling them

*Scoring*: `GroundTruthScorer`s give each role a ground-truth value - by construction (the stance an
agent was told to argue: `StanceValue`) or by measurement (hidden tests, engine centipawn loss,
resolution, a stronger-model audit `ModelAudit`, manipulation checks `PositionFollowed`).

*Sampling*, in increasing adversary strength:

* **Instructed arms / ASD** (`samplers.arms`), binary or graded over all answers.
* **Best-of-N self-play** (`samplers.pools`): any mechanism is expanded into a *sampled game tree*
  (pools of K candidates at each decision, replayed with memoized prefixes), and exact Bo(n) or
  KL-tilted policies are computed for every role by backward induction - the general form of the
  nested Bo(n)-proposer / Bo(m)-critic procedure of the "debate with self-play best-of-N" post,
  including simultaneous moves. Outputs the 1D "worm" curves and 2D meshes.
* **Prompt optimization** (`samplers.prompt_search`): an optimizer LLM that knows the mechanism (rules,
  rewards, the agent's view) proposes strategies scored by mechanism reward only (OPRO-style,
  GEPA-style reflective, evolutionary, autoresearch-style research log with a ratchet, and the reference
  GEPA optimizer itself via `integrations.gepa`). **Directives** ("honest only", "deceptive only", "sandbag",
  "collude", custom) sample a spread of strategies; the gap between the best honest and best deceptive
  strategy (held-out, paired CI) is a worst-case IC measure. Measured ground truth guards against
  "deceptive" strategies that became honest.
* **Multi-agent optimization** (`samplers.psro`, `games`): best responses against the current
  meta-equilibrium, with the optimizer told the others' strategies - the explicit version of "one
  optimizer's reasoning depends on how it models another's". Empirical games give equilibria, their
  ground-truth welfare bounds, and basins.
* **RL** (`integrations.rl`): step-wise multi-agent environments and TRL-style reward functions, with
  ground truth logged alongside reward; plus exact learning dynamics on empirical games.

The connection between these is exact (docs/theory.md §2): ASD is the first-order effect of
KL-regularized RL on ground truth, best-of-N and tilting are finite-pressure points on the same path,
and Goodhart is the covariance turning negative along it.

## 3. Beyond accuracy, toy settings and simulated gaps

* The primary outputs are incentive measures, not judge accuracy (which is still reported, as the
  control measure).
* Settings with *real* capability or access gaps: chess (engine-backed experts vs. LLM judges; Lichess
  CC0 puzzles; engine ground truth), text-to-SQL over a private database, code with hidden tests,
  Lean statement faithfulness (the proof checker settles validity, so the open question is whether the
  formal statement means the English one - judged against miniF2F's reviewed formalizations and
  single-edit mutants), forecasting with delayed resolution, QuALITY's hidden passage, plus
  ControlArena's agentic settings via the bridge. Chemistry and security, the remaining row of the
  proposal's table, are left out as dual-use.
* **Real work is stateful**, and the protocols stop looking like Q&A. `core.state` gives episodes a
  forkable, content-addressed state (files, databases, mail, a card ledger, hidden environment state);
  honest and deceptive agents act on it in separate branches (best-of-N gets a branch per sample);
  reviewers see a dossier of diffs and can be given read access to query the result; claims about the
  work are verified by executing them on the claimant's own result. Two stateful domains ship: repository
  work with hidden tests (faking success by hard-coding, editing tests, subtle bugs, doing nothing) and a
  simulated software company (burner sign-ups on the company card, fabricated rows, false claims in
  outreach, denied refunds, redefined metrics), each audited by the environment rather than by the
  instruction alone. `samplers.paired` runs the proposal's paired-branch design, including "work once,
  review many".
* **Verified claims** are systematic: a `VerificationPolicy` per mechanism chooses verifiers (quotes,
  chess legal lines - rules only, no evaluation - engine evals, SQL execution, code execution, tests,
  Lean statement structure - what a statement says, never whether it is faithful - Lean typechecking,
  model fact-checkers), whose claims are checked, the budget and the display. Every verification is
  logged, so "does ASD rise with verification access?" and "do liars make more failed claims?" are
  one-line analyses. The theory (evidence games, unraveling) says verification changes which outcomes
  are implementable at all.

## 4. Swarms and equilibrium

`mechanisms.Team` runs n agents on a shared task, then a private reporting stage checked by an auditor;
`TeamReward` (common objective) and `Whistleblower` (bounties: each / first / shared, false-report and
violation penalties, voiding the hacked score) are the two mechanisms to compare. The analytic model
(`theory.whistleblower`) makes the intuition precise: with a common reward silence is weakly dominant;
with a bounty larger than the hack's stake reporting is dominant; in between there are two strict
equilibria separated by the threshold $p^* = 1-(s/\Delta)^{1/(n-1)}$, so whether snitching pays depends
on what an agent expects others to do, training selects the equilibrium by initial propensity, and the
silent basin shrinks with team size; collective liability for unreported violations can make reporting
dominant even with small bounties. The simulated team game reproduces the analytic payoffs exactly;
`elicit_beliefs=True` records agents' predictions of their teammates so actions can be tested against
best responses; the code domain provides an LLM-scale team task with a hackable visible-test grader.

## 5. What matters besides incentive compatibility

See docs/theory.md §10: outcome quality at equilibrium, collusion robustness, equilibrium reachability
and selection, signal quality (variance, label efficiency), cost and scaling with the capability gap
and evaluation budget, exploitability of fixtures, failing loudly, participation/budget balance,
generalization, side effects (persuasion capability), measurement validity, and fair treatment of
honest uncertainty. Each has a corresponding measurement in the library.

## 6. No ground truth, or ground truth later

Items can have `status="pending"` ground truth; rewards can be deferred (market scoring rules,
resolution scores). `so_arena.release` publishes a ground-truth-free bundle - each item's mechanism
outcomes and every behaviour's rewards (what scored highly vs. lowly), with an HTML viewer - plus a
SHA-256 commitment to publish or timestamp. `resolve` later verifies the bundle, fills in deferred
rewards, scores everything and writes the report; the forecasting domain can fetch resolutions itself.
Mechanisms that need no ground truth (peer prediction with BTS or multi-task rules, market making)
are built in.

## 7. Mechanism design

docs/theory.md works this out: mechanisms as game forms with payments; four levels of IC for trained
agents; the covariance identity; equilibrium concepts and learning; implementation theory
(Maskin, Moore-Repullo) as the formal ancestor of debate and where it breaks for AI; rational proofs;
mechanism design with evidence as the theory of verified claims; costly state verification and the
result that random audits make any mechanism IC in expectation, with the mechanism's value becoming
*variance reduction* ($1/(1-\rho^2)$ audits per audit); peer prediction, BTS and markets for
ground-truth-free settings; team production, the informativeness principle and leniency programmes for
swarms.

## 8. The lab proposal

The library implements the proposal's "Strand 5" build-out list:

| Proposal item | Status |
|---|---|
| protocols as first-class objects over agentic episodes, incl. ControlArena micro-protocols | done (mechanisms; ControlArena bridge both ways) |
| paired-arm runner | done: forkable content-addressed state (files incl. git repositories, SQLite databases, mail, a card ledger, simulated counterparties with hidden state), per-decision branches, dossiers of diffs, access levels, work once / review many (`core.state`, `samplers.paired`, domains `repo` and `firm`); container isolation and Postgres/Neon/Dolt or Stripe-sandbox backends plug in as environments |
| judge interface for LLMs and humans with evaluation budget as a parameter | done (LLM policies, `HumanModel(time_budget_s=)`, `CallbackModel` for a rating UI; budget sweeps); a hosted rating UI is future work |
| metrics with uncertainty: ASD both forms, expected scores, judge accuracy, discrimination at fixed FPR, control safety/usefulness | done (PGR requires weak-to-strong training runs - compute it from their outputs) |
| model registry with parameters, compute, inference settings | done (`ModelSpec`, `samplers.sweeps.capability`) |
| hooks exporting any protocol as an RL reward | done (`integrations.rl`) |
| cost simulation before a run | done natively (`so-arena estimate`, `configure(simulate=True)`), in the style of `costly` |
| distribution as Inspect tasks and a ControlArena extension, permissive licence | done (MIT); PyPI publication is a one-command step once you want it |

Other proposal items it directly supports: ASD from existing ControlArena logs (the first milestone);
the chess setting with Lichess data; scaling grids and breakdown gaps; evaluation-time (Kinniment)
curves; RL-vs-prompted validity tests; honest/deceptive paired work review on firm-like tasks with the
reviewer's access (report, diffs, queries) as a protocol dimension.

Two notes on the ASD paper: the Brier-form ASD here uses SOlib's code convention,
$-\sum_a(\mathbf 1[a=\text{pos}]-p_a)^2$ per agent, i.e. $2[(1-p_\bot)^2-(1-p_\top)^2]$ for binary
questions (so a perfect judge scores 2); and every built-in mechanism shows each agent its assigned
answer (a regression test guards against the "Propaganda agent never told its side" class of bug).
