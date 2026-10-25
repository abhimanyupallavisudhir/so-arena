# OversightArena (`so-arena`)

**A library for measuring what scalable-oversight mechanisms incentivize.**

A scalable-oversight *mechanism* is any oversight workflow (a control protocol, a debate, a review
pipeline, a monitored agent, a team of agents) together with a *reward rule* that scores each agent,
so the mechanism can be used for training. What matters is not only whether the overseer reaches the
right answer today, but whether **optimizing the mechanism's rewards pushes agents toward good
behaviour** - and keeps doing so under training. OversightArena makes that measurable:

* every episode records the **mechanism's reward** for each agent *and* the **ground-truth value** of
  what the agent did (known to the experimenter, hidden from the mechanism);
* **samplers** produce behaviour at increasing optimization pressure - instructed arms (ASD),
  best-of-N/KL-tilted pools on sampled game trees, directive-constrained prompt search (built-in
  algorithms or GEPA), PSRO, RL;
* **metrics** relate reward to ground truth - ASD (log/Brier/graded), pairwise concordance, label
  efficiency, optimization curves, equilibria and their ground-truth welfare;
* **settings with real capability gaps** - chess (engine ground truth), text-to-SQL over a private
  database, code with hidden tests, Lean statement faithfulness, forecasting with delayed resolution,
  QA with hidden passages - with **verified claims** as a first-class experimental axis;
* **stateful work** - honest and deceptive agents act on forked copies of the same repository or company
  (database, mail, payments), and reviewers with different access decide whether to accept the result.

It builds on [Inspect](https://inspect.aisi.org.uk) for models and interoperates with
[ControlArena](https://control-arena.aisi.org.uk): ControlArena logs become rewarded episodes (ASD next
to safety/usefulness), and any SO-arena mechanism can run as a ControlArena monitor.

![Optimizing a proposer against a single judge vs. with a critic](docs/figures/bon_curves.png)

*Offline demo (synthetic world): best-of-n optimization against a single judge raises the proposer's
reward while its accuracy collapses; with a critic, the same optimization raises accuracy.*

## Install

```bash
pip install -e ".[dev]"          # core + Inspect + python-chess + test tools
pip install -e ".[all]"          # + ControlArena
```

Python ≥ 3.11. Chess ground truth uses Stockfish (`apt install stockfish`).

## Quickstart

Offline demos (no API keys):

```bash
so-arena demo            # writes runs/demos/{asd,optimization,swarm,work,monitoring,release,...}/report.html (chess: needs Stockfish)
so-arena run configs/asd_synthetic.yaml
```

A real experiment - the ASD setup with a strong agent and a weak judge:

```python
import so_arena as soa
from so_arena.domains import get_domain
from so_arena.mechanisms import DirectJudge, Propaganda, Consultancy, Debate
from so_arena.samplers.arms import ASDExperiment

soa.configure(cache_dir=".cache/so_arena")
dom = get_domain("gsm8k")
exp = ASDExperiment(
    [DirectJudge(), Propaganda(), Consultancy(rounds=2), Debate(rounds=2)],
    dom.load(limit=100),
    agent=soa.LLMPolicy("anthropic/claude-haiku-4-5"),
    fixtures={"judge": soa.LLMPolicy("openai/gpt-4o-mini", elicitation="logprobs", temperature=0)},
    ctx=dom.context(), store="runs/asd-gsm8k")
exp.run()
print(exp.summary())
```

Estimate cost before spending anything: `so-arena estimate configs/asd_gsm8k_llm.yaml` (a dry run with
simulated models that prices every call).

## What's in the box

| | |
|---|---|
| **Mechanisms** | naive judge, propaganda, consultancy, debate (simultaneous/sequential, k debaters, cross-examination, judge questions, zero-sum), comparisons (the RLHF / reward-model signal), prover-verifier games, reviewed work (single judge, proposal→critique→rebuttal, dossiers of state changes), work-then-review, monitored work (action / CoT / tool-call / probe monitors, fail-closed scores, defer-to-trusted, simulated fixed-AUROC probes), teams with whistleblowing (partial observation, covert team channels, shared repositories with reverts), market making, prediction markets (LMSR), forecasts with immediate judge ratings, peer prediction (BTS, multi-task, correlated agreement, DMI), confessions. Write your own as plain async Python. |
| **Reward rules** | judge-score transforms, zero-sum, preference scores, prover rewards, audited proper scores for trainable judges, team reward, whistleblower bounties (with leniency, misprision and collective liability), monitor penalties, random audits (penalty or inverse-probability-weighted), resolution scores, market scoring rules - composable and re-applicable to logged episodes |
| **Samplers** | instructed arms (ASD), behaviour arms on stateful tasks (paired branches), sampled game trees with exact best-of-N / tilting backward induction, prompt search (OPRO / reflective / evolutionary / autoresearch-style ratchet, or GEPA) with honest/deceptive/… directives, parameter search over programmatic agents, PSRO (Nash, fictitious play or level-k meta-solvers), RL environments and DPO preference pairs |
| **Metrics** | ASD (any proper score, paired bootstrap), graded ASD, concordance, label efficiency, incentive gap recovered, expected scores, ground-truth regret, judge accuracy, AUROC/TPR; diagnostics: reward SNR, length/slot/label biases, calibration, compliance, cost |
| **Game theory** | empirical games, pure/strict/mixed Nash, zero-sum values, (coarse) correlated equilibria with ground-truth welfare bounds, replicator dynamics, basins, natural and vanilla policy-gradient learning dynamics (exact, or on sampled episodes), an analytic whistleblower model with partial observation, monitor-penalty thresholds (obfuscation vs. deterrence) |
| **Domains** | synthetic persuasion, team, monitoring & forecasting worlds, HiddenBits disclosure (exact Bayesian judges), chess, text-to-SQL, code with hidden tests, Lean statement faithfulness (miniF2F), forecasting, GSM8K, MMLU, TruthfulQA, GPQA, QuALITY; stateful: repository work with hidden tests, a simulated software company |
| **Stateful work** | forkable content-addressed state (files, SQLite, mail, ledgers, hidden environment state), a branch per decision and per best-of-N sample, reviewer dossiers of diffs, read/write access per role (and per decision), trusted tool-call records and agent-editable action logs, paired-branch experiments (work once, review many) |
| **Verified claims** | `<claim kind="...">` checked by domain verifiers (quotes, chess legal lines, engine evals, SQL execution, code execution, tests, Lean statement structure, Lean typechecking, fact checkers, hidden bits) under per-mechanism claim or cost budgets, verification noise (forged-but-realistic errors), and rules for who sees verdicts and how |
| **Deferred ground truth** | release mechanism results with a SHA-256 commitment, or sealed (per-item commitments and Merkle proofs, revealed later); resolve and score when the truth arrives |
| **Infrastructure** | Inspect backend (all providers), response cache, dry-run cost simulation, model registry, resumable run stores, YAML specs (any registered reward rule, verifier or scorer), CLI, self-contained HTML reports, a browser rating interface for human judges with time budgets, a filesystem sandbox that keeps agent code away from hidden state and hidden tests, Inspect task export, ControlArena bridge, GEPA backend for prompt search |

## Documentation

* [docs/design.md](docs/design.md) - architecture; how rewards, ground truth and ControlArena's scores relate
* [docs/theory.md](docs/theory.md) - scalable oversight as mechanism design: the covariance identity linking
  ASD, best-of-N and KL-regularized RL; equilibria; implementation theory; verified evidence; audits as
  control variates; monitoring and obfuscation; peer prediction and markets; teams and whistleblowing; what else matters besides IC
* [docs/experiments.md](docs/experiments.md) - cookbook for every experiment family
* [docs/extending.md](docs/extending.md) - adding domains, stateful environments, verifiers, tools, scorers, mechanisms
* [docs/beyond_ic.md](docs/beyond_ic.md) - measuring what matters besides incentive compatibility
* [docs/controlarena.md](docs/controlarena.md) - how ControlArena's protocols and scores map onto mechanisms; the bridges
* [docs/reference.md](docs/reference.md) - API reference (generated by `scripts/gen_reference.py`)
* [docs/answers.md](docs/answers.md) - how the design answers the questions in the original brief

## Gallery (offline demos)

| | |
|---|---|
| ![ASD](docs/figures/asd.png) | ![mesh](docs/figures/bon_mesh.png) |
| ASD across protocols; verified claims help most | both players optimizing (proposer × critic best-of-n) |
| ![threshold](docs/figures/snitch_threshold.png) | ![training](docs/figures/snitch_training.png) |
| when reporting a teammate's hack takes over | training selects the equilibrium by starting propensity |
| ![observation](docs/figures/snitch_observation.png) | ![npg](docs/figures/snitch_npg_vs_reinforce_high.png) |
| partial observation: below b_R a bounty buys nothing | the training algorithm selects the equilibrium too |
| ![access](docs/figures/asd_by_access.png) | ![monitoring](docs/figures/monitoring_misbehaviour.png) |
| stateful work: which faked successes each level of reviewer access exposes | penalizing an evadable monitor buys obfuscation; a fixed-AUROC probe deters |
| ![disclosure](docs/figures/hiddenbits_propaganda.png) | ![bon budget](docs/figures/bon_budget_lies.png) |
| verified claims: only a sceptical judge makes a lone advocate's disclosure unravel | best-of-N moves a liar's lies beyond the verification budget |
| ![released](docs/figures/release_proxy.png) | ![resolved](docs/figures/release_resolved.png) |
| released before resolution: a judge's rating rewards extreme forecasts | resolved: the proper log score penalizes them |
| ![chess honest](docs/figures/chess_honest.png) | ![chess cherry-picked](docs/figures/chess_cherry_pick.png) |
| chess: honest verified lines rescue trapped shallow judges | cherry-picked (still legal) lines mislead a consultant's judge; debate recovers part |

Full demo reports (self-contained HTML): [ASD across protocols](docs/reports/asd.html) · [optimization pressure](docs/reports/optimization.html) · [swarms](docs/reports/swarm.html) · [stateful work](docs/reports/work.html) · [monitoring](docs/reports/monitoring.html) · [verified claims](docs/reports/hiddenbits.html) · [best-of-N vs. a verification budget](docs/reports/bon_budget.html) · [release now, resolve later](docs/reports/release.html) · [chess: engine experts vs. shallow judges](docs/reports/chess.html). The demos use synthetic domains or scripted agents, so they illustrate the machinery, not findings about language models.

## Tests

```bash
pytest                   # offline suite
pytest -m network        # tests that download data
pytest -m engine         # tests that need Stockfish
```

## License

MIT.
