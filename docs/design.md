# Design

OversightArena measures how scalable-oversight mechanisms *incentivize* the agents they oversee. The
library is organized around one separation and four layers.

## The separation: reward vs. ground truth

| | Mechanism reward $u$ | Ground-truth value $v$ |
|---|---|---|
| What it is | what the protocol pays each trainable agent - what training optimizes | how good the behaviour really was |
| Computed by | the mechanism, from what its fixtures (judges, graders, verifiers) can see | the experimenter, from information the mechanism lacks |
| In code | `RewardRule` -> `Episode.rewards` | `GroundTruthScorer` -> `Episode.ground_truth["role_values"]` |
| Available when | always (that's the point) | in chosen experimental settings; possibly later (`status="pending"`) |

Incentive compatibility is the relation between the two. Every metric in `analysis.metrics` is a
statement about the joint distribution of $(u, v)$ under some behaviour distribution.

### How this relates to "scores" in ControlArena

ControlArena (and Inspect) use *scorers* for two different things, neither of which is a per-agent
reward:

* **Task scorers** (`main_task_success`, `side_task_success`) - ground truth about what happened. In
  SO-arena terms these are ground-truth values.
* **Monitor scores** (`suspicion_score`, aggregated by `max_suspicion_scorer`) - outputs of an
  oversight component. In SO-arena terms these are part of the mechanism's *outcome*.

Safety and usefulness are then computed post hoc (audit the top-$k$% most suspicious runs). What is
missing for training is a *reward rule*: e.g. "the untrusted agent earns main-task success unless its
suspicion exceeds the audit threshold". `integrations.controlarena.episodes_from_control_logs` attaches
such rules to existing ControlArena logs; ASD then compares the reward of honest and attack runs of the
same task - so any ControlArena evaluation can report ASD next to safety and usefulness
(`control_report`). With the plain task reward, ASD is typically ~0 (attacks also complete the main
task): a control protocol that is safe at deployment can still give *no training incentive* against
attacking.

## Layer 1: items and domains

`TaskItem` holds the public question, answer options (with hidden values), public structured
`context`, `private` information keyed by *affordance*, and hidden `ground_truth` (possibly pending).
`item.censored()` is what mechanisms receive.

A `Domain` supplies items plus the domain's **verifiers** (trusted claim checks), **tools** (private
capabilities that create capability gaps) and **ground-truth scorers**. Built in: synthetic persuasion
and team worlds (offline, exact), chess (engine ground truth, legal-line verifier, engine tool),
text-to-SQL over a private database, code with hidden tests (including a hackable team task), Lean
statement faithfulness (miniF2F formalizations vs. single-edit mutants; a rules-only structure verifier),
forecasting (Manifold; pending resolutions), and QA sets (GSM8K, MMLU, TruthfulQA, GPQA, QuALITY with
verified quotes).

### Bespoke per domain, or general mechanisms with adapters?

General mechanisms with domain adapters - the same answer ControlArena gives with its
settings x micro-protocols product. A mechanism never parses domain structure: it renders
`item.question`, the options, and the private context a role is entitled to, and it delegates
domain-specific checking to verifiers and domain-specific capability to tools. So "debate" is one
class; "chess debate with the legal-line rule and engine-backed debaters" is
`Debate(tools={"agents": ["engine"]}, verification=VerificationPolicy(verifiers=["chess_line"]))` on the
chess domain. Domain-*specific* mechanisms remain possible (subclass `Mechanism`) when a protocol's
structure itself depends on the domain, e.g. a decomposition protocol that splits a proof into lemmas.

## Layer 2: mechanisms

`Mechanism` = game form (`roles()` + `async protocol(g)`) + `RewardRule`. Roles are `RoleSpec`s:
kind (agent, judge, monitor, grader, auditor), trainable or fixture, affordances, tools, and whose chain
of thought the role can see. Per-role overrides (`affordances=`, `tools=`, `sees_reasoning=`,
`trainable=`) let an experiment reconfigure information and capability without subclassing.

The protocol is ordinary async Python calling `g.act(role, kind=..., prompt=..., visible_to=...)`. The
`Game` runtime:

* builds each role's **view** (private info filtered by affordance, turns by visibility, others' CoT
  only if permitted) and a default prompt;
* sends a **typed request** (`text`, `choice`, `probabilities`, `score`, `json`) to the role's policy,
  so mechanisms are independent of how behaviour is produced;
* **verifies claims** (`<claim kind="...">`) under the mechanism's `VerificationPolicy` and shows other
  roles annotated text (`<verified>`, `<failed>`, `<unverified>`);
* supports **simultaneous moves** (`g.simultaneous`) so that no mover sees the others' current actions;
* can **branch**: with a `BranchController`, every decision becomes a node with a pool of $K$ sampled
  candidates; replaying the protocol under different plans (with memoized prefixes) expands any
  imperative protocol into a sampled game tree. Node keys depend only on role, phase, a deterministic
  slot counter and earlier choices, so simultaneous movers share pools (information sets).

Built-in mechanisms: `DirectJudge` (naive judge), `Propaganda`, `Consultancy`, `Debate`
(simultaneous/sequential, judge questions, zero-sum option), `ReviewedWork` (worker -> critique ->
rebuttal -> reviewer: single-judge RLAIF, the proposal/critique/rebuttal protocol of best-of-N
self-play debate, and realistic work review), `MonitoredWork` (action/CoT/tool-call/probe monitors
with suspicion penalties or audits), `Team` (swarms with private reporting, auditing, belief
elicitation), `MarketMaking`, `PredictionMarket` (LMSR, deferred resolution), `PeerPrediction`
(output agreement, BTS, multi-task) and `Confession` (a separately rewarded honesty channel).

Reward rules compose (`JudgeScore("log") + 0.5 * MonitorPenalty()`, `ZeroSum`, `RandomAudit`,
`TeamReward`, `Whistleblower`, `ResolutionScore`, `MarketScoringReward`) and most can be re-applied to
logged episodes (`rescore`) to study counterfactual reward rules without new model calls.

## Layer 3: behaviour samplers

To measure $(u, v)$ we need behaviours with a range of $v$. From weakest to strongest adversary:

| Sampler | Module | Ground truth | Reaches |
|---|---|---|---|
| Instructed arms (ASD) | `samplers.arms` | by construction (+ manipulation checks) | what agents do when told |
| Natural pools, best-of-N / tilting over game trees | `samplers.pools`, `analysis.optimization` | measured | re-weightings of the base policy |
| Prompt search with directives (OPRO / reflective / evolutionary) | `samplers.prompt_search` | measured | behaviour far from the base policy |
| PSRO with prompt-search best responses | `samplers.psro` | measured | approximate equilibria |
| RL | `integrations.rl` (+ your trainer) | measured | trained behaviour |

Policies (`core.policy`): `LLMPolicy` (any Inspect provider; verbalized / logprob / vote elicitation;
private CoT; text-protocol tool use), `ScriptedPolicy`, `FunctionPolicy`, `FixedPolicy`,
`ExternalPolicy` (driven by an RL loop), `BestOfNPolicy`, `MixturePolicy` (meta-strategies).

## Layer 4: analysis

* `analysis.metrics`: ASD (log/Brier/any transform, paired cluster bootstrap), graded ASD, pairwise
  concordance, within-item correlation, label efficiency, incentive gap recovered, expected agent/judge
  scores under softmax propensities, judge accuracy, AUROC / TPR at FPR.
* `analysis.optimization`: exact Bo$n$ (unbiased and plug-in) and tilted selection, KL, backward
  induction on sampled game trees with simultaneous stages, optimization grids.
* `games`: normal-form games, pure/strict/mixed Nash, zero-sum values, (coarse) correlated equilibria
  with ground-truth welfare bounds, replicator dynamics, fictitious play, regret matching, basins,
  policy-gradient learning dynamics.
* `theory`: analytic whistleblower model; audits as control variates.
* `analysis.plots` / `analysis.report`: validated-palette charts with per-mark tooltips and
  self-contained HTML reports.

## Infrastructure

* **Models**: `get_model("provider/model")` (Inspect backend), `mock`, `sim/` (dry runs for cost
  estimation), `human`, named in-process models; on-disk response cache; `ModelSpec` registry with
  prices and capability metadata for scaling plots.
* **Runs**: `RunStore` directories (items, append-only episodes, metrics, figures, report), resumable.
* **Specs + CLI**: YAML experiment specs; `so-arena run | estimate | report | release | verify | resolve | list | demo`.
* **Releases**: ground-truth-free bundles with a SHA-256 commitment; later resolution and scoring.
* **Integrations**: Inspect tasks (`integrations.inspect_task`), ControlArena (logs -> episodes,
  mechanisms as monitors, a sandbox-free toy setting), RL (step-wise `MechanismEnv`, TRL-style reward
  functions, rollout prompts).
