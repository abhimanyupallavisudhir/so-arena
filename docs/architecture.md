# Architecture and extension contracts

## Separate the three objects being studied

1. **A task** supplies a public problem, participant-specific information, and an evaluator-only oracle. `Task.provenance` records dataset version, license and gap type. Search/training requires explicit train/test splits.
2. **A mechanism** supplies a workflow, observations and per-agent utilities. Its `run(Context)` returns `Outcome(rewards, output)`. It never receives the oracle through the runtime. `Workflow` wraps an arbitrary async function; protocol templates are conveniences, not the abstraction boundary.
3. **An evaluation** supplies independently defined behavioral metrics and task outcomes. `Scorer(Task, Episode)` executes after the mechanism. Evaluation revisions cannot change the original reward or transcript.

`Reward.value` is the scalar an optimizer maximizes. `Reward.components` records a decomposition without silently summing it. Negative and unbounded finite utilities are allowed; NaN and infinity are rejected. Record units, normalizations and reward-rule versions in `mechanism.config`. A linear rescaling preserves an exact best response but changes finite-temperature behavior and learning-rate effects.

`Evaluation.per_agent` can contain accuracy, honesty, sabotage, helpfulness, proof validity, calibration or domain-specific metrics. `outcomes` holds collective/task metrics. `quality` is a plotting convention, not an assertion that honesty equals correctness. Supply additional metrics and select them explicitly in analysis. Unknown labels are `pending` or `unavailable`, never zero.

## Participants and communication

`Role(id, trainable=True)` requires an explicit reward. Fixed judges use `trainable=False`; training a judge requires a reward rule supplied to the mechanism. The runtime makes no assumption about model strength, trust, architecture, or how many roles share one model.

`Context.ask` constructs an `Observation` with the public task, that role's private input, a filtered transcript and a deterministic call seed. It gives policies copies, so mutation of their inputs cannot alter the stored task or another participant's observation. Each `Event` has a stable within-episode id, actor, kind, payload, optional recipients, and optional channels.

A public event is visible to every participant; private events are visible only to named recipients. Reasoning or interpretability signals are separate channels and are omitted unless the recipient has an explicit `channel_access` allowlist. Absence of a channel is not evidence that the model had no hidden reasoning. Simultaneous debate uses a frozen transcript at the start of each round. It dispatches calls sequentially for reproducible ordering; this is an information constraint, not a claim about wall-clock parallelism.

Every trainable policy should be stateless with respect to calls outside its supplied observation. Stateful adapters must reset or snapshot state across interventions. Model/service global state and arbitrary Python closures are outside this API's isolation boundary.

## Tools and verification

A `Tool` has an async callback, allowed roles, version, price, description and parameter schema. The mechanism can call it directly or use `Context.interact` for a bounded JSON tool loop. A policy can request a tool but cannot grant itself access or widen the recipients. Researcher tools can expose SQL execution, chess analysis, kernel checks, hidden-test summaries, retrieval, or interpretability outputs.

Public mechanism-time verification is distinct from private post-run evaluation. A `Claim` names an author, statement, artifact, specification and scope. `verify_claim` records a tool invocation and a mechanism-created receipt containing hashes of the exact artifact/specification. A boolean written by an agent does not become a verification receipt. `receipt_matches` detects reuse against a changed claim. Receipts are provenance within the trusted runtime, not cryptographic attestations from external hardware.

A Lean kernel can establish that a proof satisfies a formal proposition without establishing that the proposition captures an English request. Represent semantic correspondence as a separate evaluation dimension. Likewise, passing hidden tests does not establish universal correctness.

## Budgets and failures

`Budget` caps model calls, tool calls, wall-clock runtime and optionally returned known costs. Model call reservations occur before awaiting, so concurrent requests cannot evade the call count. Price-unknown responses fail a configured cost budget. A returned-call check can exceed the ceiling by the cost of in-flight requests; enforce a hard spend limit at the provider or reservation layer when needed. Tools with known fixed costs are charged before execution.

Policies must report usage accurately. The core does not estimate prices from token counts. Scorer compute occurs outside the mechanism budget because it is evaluation-only; timebox expensive evaluators in their adapters and record audit/evaluation compute separately.

`run_episode` raises on malformed output or budget exhaustion. `run_campaign` collects explicit failures alongside successful records and reports completion rate. Failed runs are not assigned zero utility or silently included in incentive estimates. Persist a campaign's failure manifest using `dataclasses.asdict` when running large sweeps; `RunStore` contains completed episodes.

## Persistence and extension points

The episode id hashes the mechanism record, including task, events, utilities, configuration, seed and provenance. Evaluations are excluded from that hash and have their own checksums. `RunStore` stores immutable episode payloads and append-only evaluation history in SQLite transactions. Repeating a save is idempotent; conflicting evaluation prefixes are rejected. Save an updated episode after rescoring rather than reconstructing its earlier history.

The database is a local trusted research artifact, not a tamper-proof external ledger. JSONL exports contain complete records, including private event content; `public_result` creates a separate, explicitly labelled publication projection that omits private events, channels, oracle evidence and optimizer provenance. Public text and the final output still need ordinary researcher review before publication.

A new mechanism implements `Mechanism`; a domain supplies tasks, tools and scorers; a sampler supplies policies/interventions or conditional tree-expansion callbacks; an optimizer consumes only `Reward.value`; a trainer returns updated policies. This division supports new experiments without adding domain branches to the core. Add code/weights/model identifiers to `provenance` and callback/reward versions to configuration: Python source and model weights are not automatically fingerprinted.
