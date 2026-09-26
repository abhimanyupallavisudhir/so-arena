# Concepts and design

OversightArena evaluates **mechanisms**: protocols that assign **rewards** to the agents taking
part, so that the protocol could be used to train them. It asks whether those rewards track
**ground truth** about the agents' behaviour, under increasing optimisation pressure and in
multi-agent equilibrium. This page explains the moving parts. For the theory behind the metrics see
[theory.md](theory.md); for recipes see [cookbook.md](cookbook.md); for every public class and
function see [reference.md](reference.md).

```
             ┌──────────── Domain (≈ ControlArena setting) ────────────┐
 Task ──────▶│ TaskView (no GT) · clearances · tools · verifiers · env │
 (with GT)   └────────────────────────────┬────────────────────────────┘
                                          ▼
 Profile ──▶ Strategies (bound per task) ──▶  Mechanism.run(ctx)  ◀── Agents (LLM / programmatic / human)
 (behaviours to simulate)                       │  game form: who acts, sees, verifies
                                                ▼
                                       EpisodeRecord (transcript, outcome)
                                          │                         │
                          RewardRule ◀────┘                         └────▶ GT scorers (hidden)
                          (rewards to trainable roles)                     (per-role & principal)
                                          ╲                         ╱
                                           ▼                       ▼
                               Analyses: ASD · alignment · frontier · best-of-N · games · dynamics
```

## Mechanisms: game form + reward rule

A `Mechanism` subclass declares its **roles** and implements `async run(ctx)`, which plays one
episode through the `EpisodeContext` API:

| Method | Purpose |
|---|---|
| `ctx.ask(role, prompt, response=...)` | ask a role to act, with an optional structured response spec |
| `ctx.simultaneous({...})` | several roles move on the same state |
| `ctx.post(...)` | moderator messages |
| `ctx.positions([...])`, `ctx.assign(role, option)` | who argues for what |
| `ctx.verify(...)`, `ctx.query(gt_channel, role)` | trusted verification; declared ground-truth access |
| `ctx.set_outcome(...)` | what the mechanism concluded |

The **reward rule** is a separate object mapping the finished record to rewards for the trainable
roles. It is a pure function of the record, so episodes can be re-scored under another rule
without re-running models (`Results.rescore`). Some rules need all episodes (multi-task peer
prediction: `BatchRewardRule`) or later information (forecast resolution: `DelayedRewardRule`).

```python
import oversight_arena as oa
from oversight_arena.core.rewards import JudgeProbability

@oa.mechanism(roles=[oa.RoleSpec(name="expert", kind="expert"),
                     oa.RoleSpec(name="judge", kind="judge", trainable=False)],
              reward=JudgeProbability(transform="log"))
async def one_shot(ctx, words: int = 80):
    pos = ctx.positions(["expert"])
    await ctx.ask("expert", f"Argue for your answer in under {words} words.")
    v = await ctx.ask("judge", "How likely is each option?",
                      response=oa.ResponseSpec.distribution(ctx.task.option_ids))
    ctx.set_outcome(probs=v.data["probs"], positions=pos, decision=v.data["choice"])
```

**Roles** are *trainable* (they receive rewards and are the subject of incentive analysis) or
*fixtures* (fixed behaviour, e.g. a weak trusted judge). A judge can be made trainable
(`judge_trainable=True`) and paid through a label channel. Role *kinds* (expert, judge, monitor,
worker, critic, forecaster...) let domains assign default information access and let agent tables
route roles to models.

**Built-in mechanisms**:

| Family | Mechanisms |
|---|---|
| Judge-based | `NaiveJudge`, `Propaganda` (≈ RLHF on one argument), `Consultancy`, `OpenConsultancy`, `Debate` (k players, rounds, simultaneous/sequential, judge questions), `CrossExamination`, `OpenDebate` (the side is chosen), `ProposerCritic` (proposal → critique → rebuttal), `Comparison` (pairwise preference / RLHF), `ProverVerifier` (helpful/sneaky provers, trainable verifier) |
| Monitoring | `Monitoring` (actor → LLM/CoT monitor and/or probe → alarm/deferral; `MonitoredReward` with audits for a trainable monitor) |
| No judge | `Reporters` + `OutputAgreement`, `BTS`, `CorrelatedAgreement`, `DMI` |
| Forecasting | `Forecast` (proper scoring, delayed; or judge rating/agreement, immediate), `PredictionMarket` (market scoring rule) |
| Swarms | `Swarm` + `TeamReward` (shared objective) vs `Whistleblower` (bounties, penalties, leniency, misprision) |

### Are mechanisms bespoke per domain?

No: they are general, and domains supply adapters. This is ControlArena's answer too (settings ×
micro-protocols). A mechanism talks to the task only through the ground-truth-free `TaskView`
(question, options, info blocks, answer type, public data) and to agents only through
`ctx.ask`. The domain decides everything domain-specific:
- how a role's view is rendered (`Domain.render_task`);
- which tools a role gets (`Domain.tools`, gated by clearance);
- which claims can be checked (`Domain.verifiers`);
- whether there is a stateful environment (`Domain.make_env`: sandboxes, message boards, logs);
- what a structured submission looks like (`task.data["submit_fields"]`);
- what ground truth means (`Domain.gt_scorers`).

So one `Debate` runs on HiddenBits, GSM8K, chess, SQL, code, Lean statements or forecasting.

A mechanism that genuinely needs a domain capability declares it as an *interface*. For example,
`Swarm` requires an environment implementing `SwarmEnvironment` (apparent vs clean score,
verifiable accusations, witness logs), which `AbstractSwarm` and `SimOps` provide. That keeps
mechanisms portable across every domain that implements the interface.

## Tasks, domains and the ground-truth firewall

A `Task` holds everything known about an instance, **including ground truth**:
- option values (`Answer.value`);
- `task.gt` (e.g. the forecasting outcome, the gold SQL query);
- privileged `resources` for trusted code (hidden bits, a database path, hidden tests).

Mechanisms and agents never see a `Task`. They see a `TaskView`, from which values, `gt` and
`_`-prefixed metadata are stripped. Ground truth reaches an episode only through:
1. **the harness**, when it binds a *simulated behaviour* (a strategy with a stance: "argue for the
   correct / an incorrect option") to a task. The resolved target is private to the role playing
   it; mechanisms see only positions (`ctx.position`), never stances or honesty tags;
2. **declared ground-truth channels** (`Auditor`, `Label`, `SimulatedProbe`, `Resolution`,
   `SwarmAudit`), whose every use is logged on the episode with its cost (`record.channels`).

**Information asymmetry** is explicit:
- info blocks carry an `access` level (`"public"` or a clearance name);
- roles hold clearances (domain defaults by role kind, experiment overrides, or grants from a
  strategy such as a ControlArena attack policy being told its side task);
- tools belong to clearance groups;
- privileged structured resources are released to cleared roles through `obs.private`.

Each role's observation is sanitised for it. Other roles' private reasoning, unshared tool
traces, evidence it may not see and harness-private data are removed. This applies to
programmatic agents too. Mechanisms decide who sees what: `visible_to` for messages,
`reasoning_visible_to` for chains of thought (e.g. a CoT monitor), `tools_visible_to` for
tool calls and outputs (e.g. an action monitor), and `data_visible_to` for the structured fields
of a response (e.g. a programmatic actor's action label, which its monitor must not read).

**Provenance.** In rendered transcripts, text written by the system is marked with reserved
brackets: speaker labels (`⟦Debater A⟧:`), moderator lines, trusted tool output and verification
verdicts (`⟦VERIFIED by sql_executor⟧`). Participant-written text (messages, reasoning, tool
arguments and results, claim text, and tool output relayed from ControlArena) cannot contain those
brackets, look-alikes of them (including `[[ ]]`), or invisible and direction-changing
characters, and its continuation lines are indented, so it cannot pass itself off as system text.
LLM agents are told what the brackets mean.

**Built-in domains**, grouped by the kind of gap between experts and the judge:

| Domain | Gap | Ground truth |
|---|---|---|
| `HiddenBits` | information (synthetic; exact theory) | hidden bits |
| `QuALITY` | information (story access) | dataset labels |
| `GSM8K` | tool use (calculator) | dataset labels |
| `HFMultipleChoice`, `MMLUPro`, `GPQA` | knowledge / expertise | dataset labels |
| `ChessMoves` | **search depth** (engine vs shallow judge) | Stockfish at fixed depth |
| `PrivateSQL` | **private database access** | execution of a gold query |
| `HiddenTestsCode`, `CodeProposal` | **code execution** | hidden test suites (EvalPlus) |
| `MiniF2F` | **formal proof checking** vs meaning | faithfulness by construction; kernel check |
| `ManifoldForecasting`, `FileForecasting`, `SyntheticForecasting` | **time** (delayed truth) | question resolution |
| `AbstractSwarm`, `SimOps` | multi-agent misconduct | environment logs, hidden tests |
| `ControlArenaDomain` | ControlArena settings | the setting's own scorers |

## Verified claims

Agents can embed checkable claims in messages, e.g. `<quote>…</quote>`, `<calc>12*7 = 84</calc>`,
`<run>impl_A([3,1])</run>`, `<sql>SELECT …</sql>`, `<line>Qxf7+ Kh8</line>` or `<lean>…</lean>`.
A mechanism's `EvidencePolicy` decides:

| Field | Effect |
|---|---|
| `verifiers` | which domain verifiers are enabled |
| `roles` | whose inline claims are checked (default: every role except judges, whose claims would give them the experts' capabilities, e.g. SQL queries) |
| `budget` | verification cost per role per episode |
| `noise` | probability the verifier errs: a verdict is flipped, or an informational output (query rows, program output) is replaced by a plausible wrong one. Agents see the erring output (each verifier's `forge()` makes it look genuine); the truth is kept for ground-truth scoring |
| `show_to` | who sees verdicts |
| `requests` | roles that may call `verify_*` tools themselves, e.g. a judge |
| `share_tool_results` | show a speaker's trusted tool outputs to others |
| `annotate_unchecked` | mark claims over budget as UNCHECKED |

Only trusted code can mark a claim as verified: status marks that agents write themselves are
stripped (look-alike letters included). Verifiers check what the claim means, not only that it
runs: quotes must match whole words, chess lines may not pass (null moves), `<run>` / `<assert>`
expressions may not rebind or patch the implementations they are about, and `<lean>` snippets may
not use escape hatches (their axioms are audited). A claim the verifier cannot evaluate is
unverified, not refuted: `<calc>` reads common notation (`12 × 7`, `12 x 7`, `$12`, `50%`,
`1,000`) and reports anything else as unchecked. Domain knobs add verifier *strength*, e.g.
`ChessMoves.verify_depth` (0 = legality only) or the quote length limit. Together these make "what can be proven, by whom, how cheaply, how
reliably" a systematic experimental factor (see `sweep` and the `hiddenbits` demo).

## Behaviours: strategies, profiles and samplers

A `Strategy` specifies how a role behaves:
- private instructions (for LLMs), with `{target}` / `{target_text}` placeholders;
- an optional **stance** resolved against ground truth by the harness (`CORRECT`, `INCORRECT`,
  `OPTION`, or `FREE`);
- `params` for programmatic agents and sampling (temperature...);
- `tags` (e.g. `honest`) and optional `clearance` grants.

A `Profile` assigns strategies (and sample seeds, per step if needed) to roles. **Profile sources**
generate the profiles to run on each task; they are the samplers at the experiment level.

| Source | Samples |
|---|---|
| `Stances` | ASD worlds: each listed role argues the correct or an incorrect option, counterbalanced; every option for more than two options |
| `Seeds` | $n$ samples of a base policy: best-of-$n$ pools |
| `GameTree` | nested samples for sequential protocols (proposals × critiques × rebuttals) |
| `Cartesian` | all combinations of per-role strategy sets: empirical games |
| `MapProfiles`, `ProductProfiles`, `Fixed`, `FunctionProfiles` | composition and custom logic |

Adaptive samplers live in `oversight_arena.elicitation`:
- strategy libraries: argument archetypes (fabricate, cherry-pick, obfuscate…) and artifact
  archetypes (sandbag, subtle bug, backdoor, tamper…);
- prompt optimisation: `PromptOptimizer` with LLM or parameter proposers, steering
  (honest-only / deceptive-only), optional ground-truth acceptance constraints, and held-out
  evaluation;
- `AgenticOptimizer`;
- multi-agent optimisation: `PSRO` with level-$k$, fictitious-play and Nash meta-solvers, and
  opponent-aware proposers;
- RL: `StrategyGradient` (REINFORCE or natural gradient over strategy populations),
  `MechanismEnv` (a text environment for any RL stack), TRL-style `reward_function`,
  `preference_pairs` for DPO.

## Agents and models

An `Agent` maps an `Observation` to an `Action`:
- `LLMAgent`: any model Inspect supports; tool loops; scratchpads recorded as private reasoning;
  structured answers parsed strictly with retries; judge probabilities from verbal reports,
  token logprobs or sampling;
- programmatic agents in `oversight_arena.sim`: exact Bayesian bit judges, engine-backed chess
  experts and judges, forecasters, swarm workers;
- `ScriptedAgent`, `ConstantAgent`;
- humans: `WebHumanAgent` (a local web page where a person plays any role, e.g. a human judge;
  claims render as verified/refuted/unchecked chips) and `HumanAgent` (console);
- inside Inspect, LLM agents bound to model roles.

An `AgentTable` routes roles to agents by exact name, glob (`debater_*`), kind (`kind:judge`) or
`*`. Model calls are cached on disk, keyed by model, messages, config and sample index, so
best-of-$n$ pools and game trees reuse shared prefixes and re-runs are free. Distinct roles and
seeds get independent samples.

## Episodes, records and ground-truth scoring

`run_episode` binds strategies, builds the context (views, clearances, tools, verifiers,
environment), runs the mechanism and returns an `EpisodeRecord`:
- the transcript, with per-entry visibility, private reasoning, evidence and tool traces;
- the outcome;
- rewards;
- ground-truth scores;
- ground-truth channel uses;
- usage (tokens, cost), the environment's final state, and errors.

**Ground-truth scorers** (`GTScorer`) return per-role values ("how good was this role's behaviour?")
and principal-level values under `_outcome*` keys ("how good was the result?"):

| Kind | Scorers |
|---|---|
| Generic | `TargetCorrect`, `TargetValue`, `DecisionCorrect`, `JudgeProbCorrect`, `AcceptCorrect`, `ClaimAccuracy`, `StrategyTag` (intended behaviour, a weak GT), `EnvGT`, `FunctionGT` |
| Domain | bit honesty, code / SQL / proof correctness, forecast log and Brier scores, swarm cleanliness and reporting, ControlArena main/side-task success |
| Model-based ("expensive") | `LLMGroundTruth` (a privileged model rates honesty against the full hidden information), `LLMCompliance` (did a simulated behaviour do what it was told?) |

Scorers can be added or re-run after the fact (`Results.regt`), and pending ground truth can be
attached when it arrives (`Results.resolve`).

## Experiments and results

`Experiment(domain, mechanisms, agents, profiles)` runs the grid tasks × mechanisms × profiles ×
repeats concurrently:
- appends records to `episodes.jsonl`;
- resumes by deterministic episode keys covering mechanism config, task content, strategies,
  agents (for scripted agents: their code, closures, bound arguments and the globals they read),
  seed, domain config and clearances;
- recomputes ground truth on resumed records when a scorer is new, has failed, or has changed
  (scorers are fingerprinted by configuration and code).

Trusted randomness (random audits, simulated probes, verification noise, tie-breaks between
reporters) uses **common random numbers**: a draw depends on the task, the episode seed and how many
draws came before, not on the mechanism or strategies, so compared arms see the same audits.

`Results` gives long (`df`) and wide (`episodes_df`) tables, re-scoring, ground-truth resolution and
persistence. `sweep` runs an experiment factory over a parameter grid. YAML configs and the CLI
(`oversight-arena run config.yaml`) build the same objects from a registry
(`oversight-arena list`).

## Analyses

| Question | Function |
|---|---|
| Does reward favour good behaviour? | `ic.asd`, `ic.alignment` (Spearman, pairwise accuracy/AUC, slope), `ic.eas_ejs` |
| What would a strong optimiser pick? | `ic.frontier` (frontier gap, GT of the argmax, GT regret) |
| One step of optimisation | `bon.bon_curve` (exact best-of-$n$, with KL), `bon.tree_value` / `tree_mesh` (nested min-max best-of-$n$ on game trees) |
| Equilibria | `games.EmpiricalGame`: Nash (support enumeration / replicator), regret, exploitability, `outcomes()`, zero-sum value, `replicator`, `fictitious_play`, basins |
| Beyond IC | `diagnostics`: outcome accuracy and calibration (task-clustered CIs), reward signal-to-noise, slot / verbosity / label biases, compliance, cost |
| Reporting | `plots` (optimisation-pressure curves, bars, heatmaps, replicator fields, regime maps), `report.html_report` (self-contained HTML with transcript browser) |

Analyses group by `mechanism`. Configurations that share a display name (e.g. two unlabelled
`Debate` variants) get a short hash suffix in `Results.df()`, so they are never pooled, and
`EmpiricalGame.from_results` refuses to pool several configurations into one game.

## Releases

`release.create_release(results, dir)` publishes what the *mechanism* concluded (decisions,
probabilities, rewards, strategy labels, optionally transcripts) with no ground truth, as JSON plus
a static `index.html`:
- every item is hash-committed under a Merkle root (publish it to timestamp the release);
- for episodes whose strategies have stances (e.g. `argue_incorrect`), nothing identifies the
  strategies: no names, ids, hashes or timestamps, items in random order, and agent ids only when
  they do not differ between strategies. A salted commitment to the names goes in each item; the
  names stay private (`<release>.private/`) until their tasks resolve (`names="always"` overrides);
- `sealed=True` publishes only salted commitments, keeping contents private until
  `reveal_release`;
- `resolve_release(dir, resolved_tasks)` attaches ground truth once it exists (e.g. forecasting
  questions resolve), verifies the commitments and computes the incentive and accuracy metrics
  retroactively.

## Integrations

- **Inspect**: `integrations.inspect.oversight_task(...)` turns any experiment into an Inspect
  `Task` (`inspect eval`, model roles `trusted` / `untrusted`, Inspect View), and
  `results_from_logs` reads the logs back.
- **ControlArena**: see [controlarena.md](controlarena.md). The bridges cover ControlArena logs →
  Results with reward rules, settings → domains, and mechanisms → micro-protocols.

## Extending

| To add | Subclass / use | Hook |
|---|---|---|
| a mechanism | `Mechanism` or `@oa.mechanism` | `roles()`, `run(ctx)`, optionally `brief()` / `incentive_text()` |
| a reward rule | `RewardRule` (`BatchRewardRule`, `DelayedRewardRule`) | `__call__(record)` |
| a domain | `Domain` | `load()`, optionally `render_task`, `tools`, `verifiers`, `make_env`, `gt_scorers` |
| a verifier | `Verifier` | `verify(claim, env)`, optionally `forge()` for noise |
| a ground-truth scorer | `GTScorer` | `score(task, record)` or async `ascore` |
| an agent | `Agent` | `act(obs)` |
| a probe | `Probe` / `FunctionProbe` | `query(ctx, role)` |
| a profile source | `ProfileSource` | `profiles(task, roles)` |

Register components for configs with `registry.register(kind, name)`.
