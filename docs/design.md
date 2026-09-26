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
capabilities that create capability gaps) and **ground-truth scorers**. Built in: synthetic persuasion,
team, monitoring and forecasting worlds (offline, exact), chess (engine ground truth, legal-line verifier, engine tool, engine-backed advocates and minimax judges of dialable depth),
text-to-SQL over a private database, code with hidden tests (including a hackable team task), Lean
statement faithfulness (miniF2F formalizations vs. single-edit mutants; a rules-only structure verifier),
forecasting (Manifold; pending resolutions), QA sets (GSM8K, MMLU, TruthfulQA, GPQA, QuALITY with
verified quotes), and two *stateful* domains: repository work with hidden tests and a simulated software
company (see "Stateful work" below).

Items must not be answerable without reading them. Generated options and mutants are built so that
position and shape carry no signal (GSM8K options are an evenly spaced run with the answer at a random
place; code and Lean mutations come in pairs that add and remove structure), and blind baselines report
what rules that never read the question would score (`domains.code.blind_baselines`,
`domains.qa.blind_baseline`) - compare a judge against them, not only against chance. Chess `eval_claim`
items choose their positions for the whole set (`domains.chess.balanced_eval_claims`): within every
combination of check, material, mobility, last capture and side to move there are as many "yes" as "no"
items, so rules reading those score chance (a logistic model on twenty such board features scores about
0.58, down from 0.83 when each puzzle's position was chosen on its own). Chess `which_move` keeps a weak
tell ("pick the move that gives check" scores about 0.59): about 70 puzzles have no suitable quiet
alternative.

### Bespoke per domain, or general mechanisms with adapters?

General mechanisms with domain adapters - the same answer ControlArena gives with its
settings x micro-protocols product. A mechanism never parses domain structure: it renders
`item.question`, the options, and the private context a role is entitled to, and it delegates
domain-specific checking to verifiers and domain-specific capability to tools. So "debate" is one
class; "chess debate with the legal-line rule and engine-backed debaters" is
`Debate(tools={"agents": ["engine"]}, verification=VerificationPolicy(verifiers=["chess_line"]))` on the
chess domain. Domain-*specific* mechanisms remain possible (subclass `Mechanism`) when a protocol's
structure itself depends on the domain, e.g. a decomposition protocol that splits a proof into lemmas.

## Stateful work: the paired-branch design

Oversight of real work is oversight of *state changes*: an agent edits a repository, writes to a
database, sends mail, moves money. `core.state` makes state a first-class part of an episode:

* A **snapshot** is an immutable, content-addressed state in a `StateStore`: a tree of agent-visible
  files (repository, SQLite databases, mailbox, card statement) plus `hidden` JSON that only the
  environment reads (simulated customers' willingness to pay, which sign-ups really came through the
  product). Identical states have identical ids, so unchanged work maps back to its parent.
* An `Environment` (`Domain.environment()`) builds each item's starting state $S_0$ from the
  *uncensored* item - hidden state may come from ground truth - and supplies tools that act on
  **workspaces** (`shell`, `read_file`, `write_file`, `sql`, `diff`, and domain tools such as
  `send_email` or `refund`).
* `RoleSpec.state_access` is `"none"`, `"read"` or `"write"` (per-role override `state_access=`). Each
  decision of a role with access runs in its own working copy, forked on first use: with `write` the
  frozen copy becomes the episode's current state; with `read` it is discarded, so a reviewer can run
  the tests without changing the work under review. In best-of-N trees every sampled candidate gets its
  own branch, so "sample N attempts at the task and keep the one the reviewer likes" is exact.
* Reviewers see a **dossier**: the report plus `g.state_diff()` - text diffs, database rows added,
  deleted and changed, appended mail and ledger records (never hidden state). Access levels - report
  only, report and diffs, querying the result - are crossed with the protocol.
* Claims about the work are **verifiable on the claimant's own result**: `CommandClaimVerifier` runs a
  command on it, `ProtectedCommandVerifier` runs trusted checks with protected files (tests, runner)
  restored from $S_0$, `QueryClaimVerifier` runs read-only SQL.
* Episodes record `initial_state` and `final_state`; ground-truth scorers audit the final state
  (hidden tests, the environment's fraud ledger) via `core.state.final_view`. Audits never trust records
  the agent could have written: the simulated firm's true books are rebuilt from its starting database
  plus the rows its tools recorded, and any other change to customers, payments or activity counts as
  tampering; the repository's original tests are judged outside the code under test, which only reports the
  values its functions return, since it could exit early, print a runner's confirmation or read its token. Every tool call is also kept in the episode (the trusted
  record), whatever the agent does to its own logs or says about its calls (a policy's own account is
  kept apart, as `metadata["reported_tool_calls"]`, when it differs).
* LLM roles with tools **investigate before deciding**: a verdict, score or probability is elicited
  after a tool loop ending in private notes, so a reviewer with read access really inspects the work
  (`LLMPolicy(investigate=True)`, the default).

`samplers.paired` runs the design: behaviour arms (honest, and deceptive in several ways) on the same
tasks, reviewed live by any work mechanism, or **worked once and reviewed many times**: `Work` produces
results, `review_items` turns each into an accept/reject item whose state is the result, and any
protocol with answer options (a single reviewer, consultancy, debate for and against acceptance)
decides it; `arm_asd`/`acceptance_asd` report ASD against each deceptive arm and its worst case.

Isolation between branches is by copying directories (fine for small workspaces). Isolation of the
*agents* from what grades them is by a sandbox (`core.sandbox`): every agent-controlled command -
workspace shells and test runs, candidate programs, executable claims, Lean checks - runs in a private
view of the file system in which the temporary and home directories are empty except for its own
working directory, everything else is read-only, the interpreter is visible, and no state store, work
root, run directory, dataset, bundled sample, response cache or the experimenter's working directory is
(`sandbox.hide(path)` adds more), with host processes invisible, no network and no capabilities. Backends are bubblewrap or unprivileged user namespaces (`unshare` + `setpriv`),
and - in containers without user namespaces - Landlock: a kernel-enforced allow-list (hidden directories inside
visible ones are granted around) plus a seccomp filter for sockets, ptrace and namespaces. Where the kernel has
Landlock it is also a second layer under the namespace backends. Where no backend exists, agent code is refused
unless the experimenter opts in (`SO_ARENA_ALLOW_UNSANDBOXED=1`); a Landlock launcher that cannot confine its
command refuses to run it rather than falling back. It is filesystem isolation for honest experiments, not a boundary
against a determined attacker. The abstraction is backend-neutral: an environment backed by containers,
git worktrees, Postgres/Neon or Dolt branches only has to implement forking and freezing.

## Layer 2: mechanisms

`Mechanism` = game form (`roles()` + `async protocol(g)`) + `RewardRule`. Roles are `RoleSpec`s:
kind (agent, judge, monitor, grader, auditor), trainable or fixture, affordances, tools, and whose chain
of thought the role can see. Per-role overrides (`affordances=`, `tools=`, `sees_reasoning=`,
`trainable=`) let an experiment reconfigure information and capability without subclassing.

The protocol is ordinary async Python calling `g.act(role, kind=..., prompt=..., visible_to=...)`. The
`Game` runtime:

* builds each role's **view** (private info filtered by affordance, turns by visibility, others' CoT
  only if permitted, others' positions once they have spoken - or earlier if the mechanism publishes
  them with `g.publish_positions`) and a default prompt;
* sends a **typed request** (`text`, `choice`, `probabilities`, `score`, `json`) to the role's policy,
  so mechanisms are independent of how behaviour is produced;
* **verifies claims** (`<claim kind="...">`) under the mechanism's `VerificationPolicy` and shows other
  roles annotated text (`<verified>`, `<failed>`, `<executed>`, `<unverified>`). *Verified* means a
  trusted checker confirmed what the claim states (an `expect` that matched, a stated goal or
  hypothesis); code, a command or a query that merely ran is *executed*, its output shown as the
  claimant's own - so printing "B passes all hidden tests" proves nothing. The markers are reserved: the
  same tags typed by an agent, in files it changed, its reasoning or tool calls - also as look-alike
  brackets or letters, HTML entities or escapes - are neutralized everywhere they are shown, so only a
  verifier can produce them. Budgets (`budget_per_role`, or a cost `budget`) count a role's verifications
  along the current path, so best-of-N siblings never share a budget; verification noise is a chance move
  shared by siblings, its errors indistinguishable from correct checks; `show_to` limits who sees verdicts;
* supports **simultaneous moves** (`g.simultaneous`) so that no mover sees the others' current actions;
* can **branch**: with a `BranchController`, every decision becomes a node with a pool of $K$ sampled
  candidates; replaying the protocol under different plans (with memoized prefixes) expands any
  imperative protocol into a sampled game tree. A decision's key is its role's **information set** -
  the request it is sent (prompt and view), its own earlier choices, and the state if it can see it -
  so nodes that differ only in moves the role cannot see (a simultaneous partner's move, another role's
  private turn, a hidden draw) share one pool, and `evaluate_tree` gives them one selection, weighted by
  how likely each node is (counterfactual reach). Selecting per node instead would let best-of-N act on
  information the role never had. Each pool's cost is charged to exactly one leaf episode, so leaves
  add up to the tree's cost;
* draws **nature's moves** (audits, tie-breaks) with `g.chance(tag)`, from the item, repeat, seed and tag
  only - never from what a role did. Every candidate of a decision faces the same draw, so best-of-N
  selects on what the audit would find rather than on audit luck, and arms are compared on the same
  audits (common random numbers). Reward-time audits (`RandomAudit`, `JudgeAuditScore`) draw the same way.

Episode ids hash the mechanism's full configuration (including its verification policy and the structure
of its reward rule, with the parameters its functions close over - an oracle's noise level, a penalty's
weight), the item's censored content, each player's policy description (for a model, its name and the
constructor arguments that change its answers, such as a `base_url`), stance and label, and the seed:
resuming a run store reuses an episode only if all of these match. Episodes whose ground-truth scoring
failed (`gt_status="error"`) are scored again when the store is resumed. Analyses label each
configuration of a mechanism separately (`debate(transform="prob")`) whenever a name covers several. Per-role overrides
(`affordances=`, `tools=`, ...) only add to a role's spec, and keys must name a role, `"agents"` or
`"all"`.

Built-in mechanisms: `DirectJudge` (naive judge), `Propaganda`, `Consultancy`, `Debate`
(simultaneous/sequential, $k$ debaters, cross-examination, judge questions, zero-sum option), `Comparison`
($k$ candidates and a judge's preference: the RLHF / reward-model signal), `ProverVerifier` (helpful or sneaky
provers drawn by chance, a trainable verifier), `ReviewedWork` (worker -> critique ->
rebuttal -> reviewer: single-judge RLAIF, the proposal/critique/rebuttal protocol of best-of-N
self-play debate, and realistic work review with a dossier of state changes), `Work` (the work stage
alone, for reviewing each result under many protocols), `MonitoredWork` (action/CoT/tool-call/probe monitors
with suspicion penalties, flags or audits, missing scores failing closed, defer-to-trusted, simulated
fixed-AUROC probes), `Team` (swarms with private reporting, auditing, belief
elicitation, partial observation of teammates' work, self-reports, team channels that overseers may or may not read, shared stateful workspaces whose violators'
changes can be reverted), `MarketMaking`, `PredictionMarket` (LMSR, deferred resolution), `Forecast` (forecasters, an optional
immediate judge rating, resolution scores), `PeerPrediction`
(output agreement, BTS, multi-task, correlated agreement, DMI) and `Confession` (a separately rewarded honesty channel).

Reward rules compose (`JudgeScore("log") + 0.5 * MonitorPenalty()`, `ZeroSum`, `RandomAudit`,
`TeamReward`, `Whistleblower`, `ResolutionScore`, `MarketScoringReward`, `PreferenceScore`, `ProverReward`) and most can be re-applied to
logged episodes (`rescore`) to study counterfactual reward rules without new model calls.

## Layer 3: behaviour samplers

To measure $(u, v)$ we need behaviours with a range of $v$. From weakest to strongest adversary:

| Sampler | Module | Ground truth | Reaches |
|---|---|---|---|
| Instructed arms (ASD) | `samplers.arms` | by construction (+ manipulation checks) | what agents do when told |
| Behaviour arms on stateful tasks (paired branches) | `samplers.paired` | by construction + audit of the final state | honest vs. several ways of faking success |
| Natural pools, best-of-N / tilting over game trees | `samplers.pools`, `analysis.optimization` | measured | re-weightings of the base policy |
| Prompt search with directives (OPRO / reflective / evolutionary) | `samplers.prompt_search` | measured | behaviour far from the base policy |
| Parameter search over programmatic agents (with measured-value constraints) | `samplers.param_search` | measured | the parameter space's frontiers |
| PSRO with prompt-search best responses | `samplers.psro` | measured | approximate equilibria |
| RL | `integrations.rl` (+ your trainer) | measured | trained behaviour |

Policies (`core.policy`): `LLMPolicy` (any Inspect provider; verbalized / logprob / vote elicitation;
private CoT; text-protocol tool use), `ScriptedPolicy`, `FunctionPolicy`, `FixedPolicy`,
`ExternalPolicy` (driven by an RL loop), `BestOfNPolicy`, `MixturePolicy` (meta-strategies).

## Layer 4: analysis

* `analysis.metrics`: ASD (log/Brier/any transform, paired cluster bootstrap), graded ASD, pairwise
  concordance, within-item correlation, label efficiency, incentive gap recovered, expected agent/judge
  scores under softmax propensities, judge accuracy, AUROC / TPR at FPR. A judgment that could not be
  parsed falls back to the uninformative answer, which pulls ASD toward 0: ASD reports the parse-failure
  rate (overall and per arm), warns above 5%, and can drop such episodes (`exclude_unparsed=True`).
* `analysis.optimization`: exact Bo$n$ (unbiased and plug-in) and tilted selection, KL, backward
  induction over the information sets of sampled game trees (fictitious play where moves are hidden or
  simultaneous, with the fixed-point gap reported), optimization grids; every value comes with its
  selection-weighted label coverage (`<key>_coverage`), and values resting on too little of the selected
  mass are dropped (`min_coverage`).
* `games`: normal-form games, pure/strict/mixed Nash (support enumeration for two players;
  `approximate_nash` for more - the lowest-NashConv profile found by replicator dynamics and by minimizing
  a function that vanishes exactly at equilibria - whose NashConv PSRO records and flags when it is only
  approximate), zero-sum values, (coarse) correlated equilibria with ground-truth welfare bounds,
  replicator dynamics, fictitious play, regret matching, basins, policy-gradient learning dynamics.
  Expected ground-truth values of profiles, like those of prompt-search strategies, come with their
  label coverage and are NaN below 50%.
* `theory`: analytic whistleblower model; audits as control variates; monitor-based rewards (when a
  penalty on alarms buys obfuscation instead of deterrence).
* `analysis.diagnostics`: what matters besides IC - reward signal-to-noise, length, slot and option-label
  biases at fixed ground truth, calibration, compliance of instructed arms, cost; `metrics.gt_regret` for
  continuous ground truth.
* `analysis.plots` / `analysis.report`: validated-palette charts with per-mark tooltips and
  self-contained HTML reports.

## Infrastructure

* **Models**: `get_model("provider/model")` (Inspect backend), `mock`, `sim/` (dry runs for cost
  estimation), `human`, named in-process models; on-disk response cache, keyed by the model's identity
  (name and answer-relevant arguments), the messages, the options and the *draw* - the role, item and
  decision making the call - so identical prompts from different roles are separate samples (also for
  backends seeded from the sample index) and a warm cache replays each role's own draw; `ModelSpec`
  registry with prices and capability metadata for scaling plots.
* **Runs**: `RunStore` directories (items, append-only episodes, metrics, figures, report), resumable.
* **Specs + CLI**: YAML experiment specs, whose reward rules, verifiers and scorers a registry builds from
  `{type: ...}` mappings (`so_arena.registry`); `so-arena run | estimate | report | release | verify | resolve | list | demo`.
* **Releases**: ground-truth-free bundles with a SHA-256 commitment; later resolution and scoring. Sealed
  releases publish only salted per-item commitments with Merkle inclusion proofs, and reveal items later.
* **Integrations**: Inspect tasks (`integrations.inspect_task`), ControlArena (logs -> episodes,
  mechanisms as monitors, a sandbox-free toy setting), RL (step-wise `MechanismEnv`, TRL-style reward
  functions, rollout prompts, same-context preference pairs for DPO).
