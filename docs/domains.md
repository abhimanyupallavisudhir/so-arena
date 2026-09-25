# Domains, capability gaps and verified claims

A protocol should consume messages, artifacts and tool results through domain adapters. Reuse the incentive contract across domains and change the environment rather than hardcoding a benchmark into a debate class.

`tasks_from_rows` loads JSON-compatible task rows with `id`, `prompt`, optional `public`, `oracle` and `role_data`. It records dataset name/version, license, split and gap type. The accepted gap labels are `capability`, `information`, `tool_access`, `cost`, `temporal` and `mixed`. A label describes the research design; it does not prove that the intended gap actually exists. Measure standalone participant competence and judge calibration under controlled access and compute budgets.

## Initial suite corresponding to the proposed settings

| Setting | Concrete support | Independent evaluation | Confound to measure |
|---|---|---|---|
| Private-database text-to-SQL | `SQLExecution`; read-only SQLite fixtures, query limits, order-sensitive or multiset result comparison | Reference execution; add human semantic review as a second metric | Database access versus reasoning competence; agreement on one fixture is not semantic equivalence |
| Forecasting | `Forecast`; pending/resolved states, negative two-class Brier scores, resolution source and revision history | A versioned outcome with timestamp/source | Outcome manipulation, resolution ambiguity, unresolved-selection bias, correlated events |
| Code with hidden tests | `ArtifactScorer` around a caller-supplied sandbox test runner; `Tool` for visible tests | Hidden-test pass rate and separate sabotage metrics | Test leakage, test quality, evaluator access, overfitting to visible tests |
| Lean proofs | `ArtifactScorer` around a pinned kernel/toolchain; `Claim`/receipt for public checks | Kernel acceptance and separate English/formal correspondence | A proof of the wrong statement; trusted axioms, toolchain/version differences |
| Chess | `ArtifactScorer` around a fixed-budget engine; `Tool` grants selected roles analysis access | Legal move, engine loss or outcome under a preregistered engine budget | Engine depth/noise, judge rating, access versus capability, search-budget mismatch |
| Chemistry or security tasks | The same artifact/tool contracts for domain computations and researcher-controlled environments | Domain validity, reproducibility and independently chosen harm/safety metrics | Inadequate proxies and verification scope; no domain benchmark is bundled |

Only the SQL, exact-match and forecast scorers ship with domain execution implemented. The other rows are usable integration boundaries, not bundled engines or claimed completed benchmarks. `examples/verified_artifact.py` demonstrates both public verification and private evaluation on the same artifact with different specifications.

## Tool and information ablations

Use `Context.interact` when a model should autonomously request tools. Give each tool an `allowed_roles` set and a version. Vary one or more of:

- same information, weaker versus stronger model;
- same model, different compute budgets;
- same model and compute, different tools or documents;
- verification forbidden, allowed at a price, or mandatory;
- reports visible to all, private to the auditor, or delayed;
- actions only, actions plus available reasoning, or actions plus interpretability output.

For a verified-claim experiment, preregister what each verifier certifies, which claims agents may submit, who sees the result, how audits are sampled, and which reward component depends on it. Bind receipts to artifact and specification, count checks and costs, and evaluate false acceptance separately. Tool exceptions are experiment failures, not evidence that a claim is false.

Keep reference answers, held-out tests, privileged database contents and future resolutions in evaluator data unless the experimental treatment intentionally grants them to a role or verifier. The runtime cannot prevent a custom callback from closing over private data: review that callback just as you review a dataset split.

## Data and reproducibility

The bundled SQL fixtures are small, MIT-licensed integration data created for this package. They test infrastructure and are not a representative benchmark. For real studies, bring a versioned dataset snapshot, license, sampling script, split hashes and task-family identifiers. Do not copy the cost/capability numbers from a proposed suite into experimental claims without measurement.

For large existing environments, reuse ControlArena's setting, sandbox and task scorer, and add an explicit reward protocol. For new domains, write a small loader plus public tools and private scorers. An external artifact evaluator receives `(agent_artifact, evaluator_data)` and returns finite metrics including `quality`. It owns process isolation, resource limits, environment pinning and any human adjudication workflow.
