# OversightArena: comparison of Task 1's four attempts

Reviewed 25 September 2026. **Attempt 2 is the strongest foundation for the library requested. Attempt 4 is the strongest compact alternative. Attempt 1 has substantial useful breadth but needs correctness repairs; Attempt 3 is a clean, narrower research core.** None has established that its mechanisms remain aligned under real LLM training.

## Scope and evidence

These are the platform's attempt numbers, **not PR numbers**. I imported each published branch, inspected implementation and documentation, ran all four test suites, built all four distributions, ran offline demonstrations, and reproduced several edge cases independently. I did not change the implementations or run paid model experiments.

| Attempt | Published commit inspected | Proposal | Shape |
|---|---|---|---|
| 1 | `9c6ba55c455835f7980679c936fb46bf18aee012` | Still in Do, awaiting a real-model pilot | `oversight_arena`; broad library, 15,205 Python source lines |
| 2 | `5f7e95fcba0043ea73524cc6298eda45e0975372` | [PR 3](https://github.com/abhimanyupallavisudhir/so-arena/pull/3) | `so_arena`; broad library plus stateful environments, 25,869 source lines |
| 3 | `76d5211d423b0e5c530b26b0ef729dc7c01a5b05` | [PR 1](https://github.com/abhimanyupallavisudhir/so-arena/pull/1) | `oversight_arena`; dependency-free core, 2,959 source lines |
| 4 | `2266343cb0f063a1bbe66e92930b63e2ff6248a4` | [PR 2](https://github.com/abhimanyupallavisudhir/so-arena/pull/2) | `oversight_arena`; dependency-free core, 2,762 source lines |

Attempts 2 and 4 received an additional instruction to review a lab-planning PDF; 1 and 3 did not. Attempt 2 also has extensive subsequent review fixes. This compares the available products, not model ability under equal time or instructions. Unpublished working changes, if any, are outside the comparison. I recovered the full textual prompt and attachment metadata; the original image was unavailable through the attachment endpoint. The PDF-specific comparison uses the attempts' checked-in analyses, not an independent re-audit of the PDF.

I weighted the prompt's central goals most heavily: a broadly useful experimental library; separate per-agent rewards and behavioral truth; meaningful optimization and strategic analysis; and practical settings beyond toy debate. Merely providing a callback that could eventually support an experiment earns less credit than an implemented, exercised experiment path. Actual neural RL was explicitly tentative in the prompt, so its absence is not a disqualifier.

## Fit to the requested work

| Desire | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---|---|---|---|
| Arbitrary workflows, rewards, trainable judges | Strong: separate protocol/reward objects; many templates and audit channels | Strongest practical coverage: reusable game forms/reward rules, audited judge/monitor rewards | Sound generic API; four templates; judge reward supplied by callback | Sound generic API; four templates; extra rewards supplied by callback |
| General scorers, samplers and ASD | Extensive domains, behavior libraries and metrics | Extensive; paired work branches, graded metrics, manipulation checks | Good generic paired interventions and task-cluster statistics; few bundled domains | Good explicit intent-versus-realized-behavior distinction, strata and multidimensional measurements |
| Best-of-N and optimization curves | Working empirical curves; nested evaluator uses a scalar root payoff with max/min directions | Most integrated: automatic game-tree expansion, general-sum payoffs, simultaneous stages, BoN and tilting | General-sum conditional trees, but caller writes expansion/scoring callbacks | General-sum conditional trees and chance nodes, but caller supplies expansion |
| Steered prompt search and multiple optimizers | LLM/parameter/agentic search; opponent-aware PSRO | Broadest: several search styles, actual GEPA adapter, PSRO and shared populations | Reward-only simultaneous/alternating search; frozen peers and held-out split enforcement | Reward-only and constrained search; joint updates, fresh trial factories and held-out evaluation |
| Actual learning | Strategy-level gradients; RL environment, TRL reward and DPO adapters | Exact finite-game policy-gradient dynamics; interactive RL/reward adapters | Sampled categorical REINFORCE, checkpointing and trainer interface | Sampled tabular multi-agent REINFORCE and trajectory export |
| Real work and capability gaps | Chess, code, SQL, Lean, forecasting, monitoring and simulated operations | Clear leader: these domain families plus forkable repositories/company state, dossiers, reviewer access and work-once/review-many | Executable SQL and forecasting; code/Lean/chess mostly external adapters | Executable SQL suites and chess components; larger domains remain adapters |
| Verified claims as experimental variables | Extensive budgets, visibility, noise and domain verifiers | Extensive domain verifiers and state-bound checks, budgets and access | Artifact/specification receipts and permissioned tools; fewer ready-made experiments | Scoped, versioned verifier registry; SQL/chess examples; fewer ready-made experiments |
| Swarms and equilibrium | Rich rewards, analytic threshold/basins, PSRO and empirical games | Richest integrated environments and game analysis, including correlated-equilibrium welfare bounds | Reporting workflow, finite games, coalition deviations and learning | Sealed reports, finite games, coalition/coarse-correlated regret and learning |
| Publish before truth arrives | Dedicated committed release/view/resolve workflow | Dedicated committed release/view/resolve workflow | Immutable episodes, append-only evaluations, public projection and reports | Immutable runs/evaluation sidecars, public export and delayed resolution |
| Beyond IC and mechanism-design theory | Broad economic connections and analytic companions | Most extensive synthesis and executable analysis | Particularly clear assumptions, alignment bound and limits of finite evidence | Particularly clear estimands, manipulation validity and measurement provenance |

All four understand the key conceptual distinction. None simply substitutes final task accuracy for the agent's incentive. All permit trainable judges; all have some useful optimization and delayed-evaluation infrastructure. Their differences are implementation depth and reliability, rather than failure to understand the main idea.

The motivating [ASD paper](https://arxiv.org/abs/2504.03731) evaluates how a protocol advantages truth-telling over deception. The [self-play BoN study](https://www.lesswrong.com/posts/hb8pv3zyAHGJpwz9F/debate-with-self-play-best-of-n-optimization) uses conditional proposal/critique/rebuttal trees. Accordingly, I gave more credit to testing behavioral incentives under conditional optimization than to reporting judge accuracy or flat sample ranking alone.

## Assessment of each attempt

### Attempt 2 — best fit and recommended foundation

Its principal achievement is **making realistic, stateful work a reusable abstraction**. `StateStore`, workspaces and environments support files, SQLite, hidden environment state and trusted tool records. A candidate can work on its own branch; reviewers can inspect a dossier, query its state or inspect code; multiple reviewers can assess the same work. This is a meaningful generalization of the library, consistent with the instruction to avoid adding bespoke machinery for each new experiment.

It also connects more of the research loop: configurable mechanisms, real domain loaders/verifiers, paired experiments, automatically expanded game trees, steered search, PSRO, game analysis, cost estimates, model registry, human rating interface and public releases. Its mechanism-design discussion addresses evidence, costly audits, equilibrium selection and collective incentives in considerably more depth than a minimal architecture note.

The tradeoff is a large new maintenance surface: nearly 26,000 source lines, no checked-in CI workflow, and substantial custom machinery for state, parsers, game solving, human interfaces and providers. The 386 passing tests provide useful evidence, especially the targeted regression tests, but cannot validate every combination. Its default shell workspaces explicitly are **not** security-isolated from the host or hidden snapshots; adversarial code experiments need a stronger environment backend. Lean mutation labels can also be wrong when mutations preserve meaning or fail elaboration; the source acknowledges this.

There are two methodological limits to repair before relying on broad claims:

- **Private information is not handled by a general information-set solver.** I supplied a hidden fair-coin tree whose two guess decisions share the same information-set key. `evaluate_tree` returned success 1.0, although a blind guesser cannot exceed 0.5. It optimizes each branch independently except for its special simultaneous-stage handling. The sampling key also includes previous decisions regardless of visibility. This does not invalidate ordinary public debate trees or supported simultaneous stages, but the claimed generality needs narrowing or a real information-set implementation.
- **Missing labels are renormalized away.** A uniform pool with rewards `[0, 1]` and qualities `[1, missing]` reports quality 1.0 without reporting coverage in the curve row. That is quality conditional on observed labels, not an identified unconditional expectation. This matters directly for the requested delayed-truth experiments. Publish probability-weighted label coverage and bounds or return unknown where appropriate.

Sources: [state abstraction and its limits](https://github.com/abhimanyupallavisudhir/so-arena/blob/5f7e95fcba0043ea73524cc6298eda45e0975372/src/so_arena/core/state.py), [tree evaluation and missing-data handling](https://github.com/abhimanyupallavisudhir/so-arena/blob/5f7e95fcba0043ea73524cc6298eda45e0975372/src/so_arena/analysis/optimization.py), [decision keys](https://github.com/abhimanyupallavisudhir/so-arena/blob/5f7e95fcba0043ea73524cc6298eda45e0975372/src/so_arena/core/game.py), [brief and lab response](https://github.com/abhimanyupallavisudhir/so-arena/blob/5f7e95fcba0043ea73524cc6298eda45e0975372/docs/answers.md).

### Attempt 1 — substantial breadth, lower confidence in correctness

This is much more than a scaffold. Its mechanism catalogue, evidence policies, real-domain adapters, monitoring experiments, strategy optimization, theory companions and seven-demo gallery address most of the original prompt. Particularly useful ideas include treating verifier reliability and access as experimental axes, comparing a fixed-accuracy probe with a behavior-sensitive monitor, and relating swarm training dynamics to an analytic model. It is the second broadest implementation.

However, I reproduced two errors that directly undermine its intended research use:

- **A best possible zero reward can lose to a negative reward.** `OptimizationTrace.best()` uses `value or NaN` while filtering candidates. Given rewards `[-1, 0]`, it returns `-1`. Zero is a normal optimum for log rewards, so this affects prompt selection and potentially downstream population search.
- **The hidden-test oracle can be spoofed without implementing the function.** `passes_hidden("print('PASSED'); raise SystemExit(0)", test, entry)` returns `True`. It trusts a success marker in the candidate process's stdout. An optimizer could therefore make the purported independent truth metric report success without correct code. Isolating the process alone does not fix this scoring error.

Its nested BoN API is also less general than the other attempts: it propagates one root payoff and chooses max/min at each level. That fits a zero-sum proposer/critic example, but not arbitrary role-specific utilities such as a trained judge plus non-zero-sum worker incentives. Missing truth is also dropped or renormalized in several optimization summaries. There is no CI workflow.

I would preserve selected mechanisms, demos and analytic companions, but would not choose this implementation over Attempt 2 without a repair pass. The bugs are localized; the concern is their position at the center of reward search and truth measurement.

Sources: [candidate selection](https://github.com/abhimanyupallavisudhir/so-arena/blob/9c6ba55c455835f7980679c936fb46bf18aee012/src/oversight_arena/elicitation/optimize.py), [hidden-test grading](https://github.com/abhimanyupallavisudhir/so-arena/blob/9c6ba55c455835f7980679c936fb46bf18aee012/src/oversight_arena/domains/code.py), [BoN and scalar game trees](https://github.com/abhimanyupallavisudhir/so-arena/blob/9c6ba55c455835f7980679c936fb46bf18aee012/src/oversight_arena/analysis/bon.py).

### Attempt 4 — best compact alternative, still substantially incomplete against the brief

Its strongest quality is careful experimental bookkeeping: distinct rewards/evaluations, exact/proxy/human/outcome provenance, audited versus intent-only ASD, fresh policy and environment factories for each trial, explicit snapshots, immutable storage, and scoped verification whose meaning is supplied by the checker. Its PDF review translates the additional research agenda into design implications while clearly identifying what remains external work. The small dependency-free core and CI make it easier to inspect and maintain.

It offers somewhat more concrete domain support than Attempt 3: SQL evaluation across multiple private fixtures and an engine-based chess scorer, as well as human-review packets. But a full stateful firm/repository environment, broad domain loaders, rich built-in reward mechanisms, PSRO populations, a human participation UI and neural training remain absent or external. Its ControlArena support primarily maps existing Inspect samples rather than providing Attempt 2's wider integration.

I reproduced a feature interaction that its tests miss: **switching debate to simultaneous mode silently skips requested tools**. With the same tool-requesting policies, sequential debate executes two tool calls; simultaneous debate executes zero and still reports completion. The sequential path uses `respond()`; the simultaneous path invokes policies directly. The swarm contribution stage uses that same single-action simultaneous primitive. This matters when verified evidence is the experimental treatment.

It is a good small platform on which to build, but calling it the broadly comprehensive library requested would overstate what ships.

Sources: [trial isolation](https://github.com/abhimanyupallavisudhir/so-arena/blob/2266343cb0f063a1bbe66e92930b63e2ff6248a4/src/oversight_arena/experiments.py), [verification scope](https://github.com/abhimanyupallavisudhir/so-arena/blob/2266343cb0f063a1bbe66e92930b63e2ff6248a4/src/oversight_arena/verification.py), [protocol tool paths](https://github.com/abhimanyupallavisudhir/so-arena/blob/2266343cb0f063a1bbe66e92930b63e2ff6248a4/src/oversight_arena/mechanisms.py), [PDF analysis](https://github.com/abhimanyupallavisudhir/so-arena/blob/2266343cb0f063a1bbe66e92930b63e2ff6248a4/docs/lab-review.md).

### Attempt 3 — clean research kernel, weakest practical breadth

This is a coherent small implementation with unusually clear theoretical limits. The distinction between paired margins, reward regret, alignment of optima and equilibrium selection is useful. The sufficient alignment bound is stated with the uniform assumptions it needs. General-sum nested BoN explicitly assumes perfect information and propagates unknown truth instead of silently turning it into complete knowledge. SQLite transactions and content-addressed episodes are strong infrastructure choices. The tests include an actual local ControlArena evaluation and import using a mock model.

The shortfall is practical completeness. Most realistic domains are adapter contracts rather than bundled environments; conditional tree expansion is left to the caller; trainable-judge rewards need user-written callbacks. Stateful policies and tools must be reset by the caller. Even the built-in debate, consultancy and monitoring templates call `Context.ask`, not the available `Context.interact` tool loop, so tool-assisted versions need custom workflows. Its monitoring default rewards only low suspicion, illustrating evasion incentives rather than providing a comprehensive task-plus-safety reward family.

I did not reproduce an equally direct central arithmetic or grading error in this attempt. That supports confidence in the inspected core, not a claim of exhaustive correctness. It remains much less of the requested finished experiment library than Attempts 1 and 2. Relative to Attempt 4, it has better demonstrated ControlArena integration and transactional persistence; Attempt 4 has better trial lifecycle support and a little more executable domain breadth.

Sources: [architecture and caller responsibilities](https://github.com/abhimanyupallavisudhir/so-arena/blob/76d5211d423b0e5c530b26b0ef729dc7c01a5b05/docs/architecture.md), [protocol templates](https://github.com/abhimanyupallavisudhir/so-arena/blob/76d5211d423b0e5c530b26b0ef729dc7c01a5b05/src/oversight_arena/mechanisms.py), [theory](https://github.com/abhimanyupallavisudhir/so-arena/blob/76d5211d423b0e5c530b26b0ef729dc7c01a5b05/docs/mechanism-design.md), [integration tests](https://github.com/abhimanyupallavisudhir/so-arena/blob/76d5211d423b0e5c530b26b0ef729dc7c01a5b05/tests/test_adapters.py).

## Verification performed

| Attempt | Tests rerun | Build | Offline execution rerun |
|---|---|---|---|
| 1 | 100 passed, 2 skipped | Wheel and sdist pass | BoN demo, figures and HTML report |
| 2 | 386 passed, 3 skipped | Wheel and sdist pass | All four demos: ASD, optimization, swarm, stateful work |
| 3 | 71 passed | Wheel and sdist pass | Full CLI demo with plots |
| 4 | 49 passed | Wheel and sdist pass | Full CLI demo with plots |

The five skips are network/Stockfish-dependent tests. The totals are **606 passed, 5 skipped**; test counts are evidence of execution, not a ranking criterion. Tests used a fresh Python 3.11.6 environment with Inspect 0.3.269, ControlArena 19.0.0, GEPA 0.1.4, NumPy 2.4.6, pandas 3.0.6 and Pydantic 2.13.5. Each attempt ran separately with its own `src` on `PYTHONPATH`; similarly named packages were not mixed. Builds used isolated build environments. This was not a clean wheel-install test or a visual browser audit.

Representative commands, from each attempt's checkout:

```sh
PYTHONPATH=src /path/to/review-env/bin/python -m pytest -q
/path/to/review-env/bin/python -m build --outdir /tmp/attempt-dist
```

The additional findings above are executable with [review_probes.py](review_probes.py), passing the four detached checkout paths. They are intentionally observations of the current behavior, not modifications or fixes to the attempts.

All completed demos are scripted or synthetic. Attempts 1 and 2 explicitly await credentials for real-model pilots. Attempts 3 and 4 describe model adapters rather than completed real-model validation. No attempt demonstrates neural weight training, general robustness to optimization, or a validated realistic capability-gap result with frontier models. A smaller honest scope statement is better than an unsupported claim, but it does not replace the missing infrastructure or empirical work.

## Recommendation

**For the prompt as written: choose Attempt 2 as the base.** It has the largest amount of relevant, working infrastructure and most directly addresses the move from toy Q&A to real work. Prioritize information-set correctness, missing-label coverage, isolated execution for adversarial code, CI, and a small held-out real-model pilot before relying on scientific results.

**For a deliberately small, easy-to-audit starting point: choose Attempt 4**, accepting that substantial library construction remains. Attempt 3 is competitive if ControlArena integration and transactional storage matter more than fresh environment factories and chess components.

**Attempt 1 ranks second on breadth, but not on readiness to trust its measurements.** Repair its reward-selection and oracle bugs before using its results; selectively port its useful mechanisms and demos if starting from 2.

Do not merge the four wholesale. They are alternative implementations with incompatible record types, execution APIs, package names and statistical conventions. A sensible combination is one core plus deliberately ported components and regression cases. No integration or merger was performed in this investigation.
