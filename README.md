# OversightArena

**ControlArena for scalable oversight.** OversightArena is a Python library for principled
experiments on scalable-oversight *mechanisms*: protocols that also assign **rewards** to the agents
taking part, and could therefore be used to train them. It measures whether those rewards track
**ground truth** about the agents' behaviour. Unlike final-task accuracy, it asks this under
increasing optimisation pressure and in multi-agent equilibrium: is the mechanism robust to
training?

- **Mechanisms = protocol + reward rule.** The library includes:
  - debate and its variants (cross-examination, open debate), consultancy, propaganda (≈ RLHF),
    pairwise comparison, proposer–critic, prover–verifier games;
  - monitoring with chain-of-thought monitors, probes and audits;
  - peer prediction (BTS, correlated agreement, DMI);
  - forecasting with proper and market scoring rules;
  - agent swarms with shared rewards or whistleblower bounties.

  Any role can be trainable or a fixture (e.g. a weak judge, or a trained one), and you can write
  your own mechanism in a few lines.
- **Ground truth never leaks.** Mechanisms and agents see a firewalled view of each task. Ground
  truth enters only through *simulated behaviours* ("argue for the wrong answer") and *declared,
  costed channels* (audits, labels, simulated probes, delayed resolution).
- **Incentive compatibility, measured at every level of optimisation.**
  - Agent Score Difference and reward–GT alignment;
  - exact best-of-n curves and min-max game trees;
  - steered prompt optimisation that maps honest vs deceptive frontiers;
  - PSRO / level-k multi-agent optimisation;
  - empirical games with equilibria, regret and collusion gains;
  - learning dynamics (strategy-level RL, replicator, basins);
  - adapters for weight-level RL.
- **Real capability gaps, not just hidden text.** Engine-backed chess experts vs weak judges,
  private-database SQL, code execution with hidden tests, Lean formal statements, and forecasting
  with delayed resolution. Verified claims are a controlled variable: what can be checked, how
  much, how reliably, by whom.
- **Release now, resolve later.** Publish what a mechanism concluded, hash-committed and with no
  verdict, before ground truth exists; score it when it arrives.
- **Theory.** A worked-out connection to mechanism design, with propositions proved and checked
  against simulation ([docs/theory.md](docs/theory.md)).
- **Fits the ecosystem.** Built on Inspect's model layer. Any experiment runs as an Inspect task.
  Bridges to ControlArena cover logs → reward-rule analysis, settings → domains, and mechanisms →
  micro-protocols.

## Install

```bash
git clone <this repo> && cd so-arena
uv venv && source .venv/bin/activate          # Python ≥ 3.11
uv pip install -e ".[dev]"                     # inspect-ai, python-chess, pytest
# optional: ".[data]" (Lichess puzzles), control-arena, trl; Stockfish for chess (apt install stockfish)
```

## A first experiment (no API keys)

```python
import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Consultancy, Debate, Propaganda
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
from oversight_arena.analysis import ic_report

verified = EvidencePolicy(budget=4)                  # each advocate may make 4 claims checked by trusted code
res = oa.Experiment(
    HiddenBits(n_tasks=60),                          # experts see hidden bits; the judge does not
    [Propaganda(evidence=verified), Consultancy(rounds=2, evidence=verified), Debate(rounds=2, evidence=verified)],
    {"kind:judge": BayesianBitJudge(trust=0.8),      # exact Bayesian judge, somewhat credulous
     "*": BitAdvocate(lie_rate=0.5)},                # advocates lie when true evidence runs out
    oa.Stances(),                                    # ASD worlds: argue the correct / an incorrect answer
).run()
print(ic_report(res.df(trainable_only=True)))        # ASD with CIs, alignment, frontier gap, GT regret
```

```bash
oversight-arena demo all              # seven demos with figures and HTML reports (see docs/demos.md)
oversight-arena theory swarm --stake 0.3 --bounty 0.2
oversight-arena list                  # registered domains, mechanisms, rewards, agents, GT scorers...
```

## With LLMs

Any model Inspect supports (OpenAI, Anthropic, Google, OpenRouter, Together, vLLM, Ollama...):

```bash
export OA_EXPERT_MODEL=openai/gpt-4.1 OA_JUDGE_MODEL=openai/gpt-4.1-nano
oversight-arena run configs/gsm8k_asd.yaml --limit 50        # ASD across 5 protocols; report.html
python examples/prompt_search.py --domain quality             # honest vs deceptive frontiers
inspect eval examples/inspect_task.py@gsm8k_debate            # the same, as an Inspect task
```

```python
res = oa.Experiment(GSM8K(limit=50), [Debate(rounds=2, evidence=EvidencePolicy())],
                    oa.llm_agents("openai/gpt-4.1", "openai/gpt-4.1-nano"), oa.Stances()).run()
```

Model calls are cached on disk, so re-runs, extended best-of-n pools and re-scoring are free.
Without `OA_*_MODEL`, examples fall back to Inspect's mock model, which smoke-tests the pipeline.

## The experiments this is built for

| You want to... | Use |
|---|---|
| Define a mechanism with any reward computation | `Mechanism` / `@oa.mechanism` + `RewardRule` ([concepts](docs/concepts.md)); re-score recorded episodes under other rules |
| Measure IC with prompted good/bad behaviours (ASD) | `Stances` + `ic.asd` / `ic_report` |
| Use richer behaviours and ground truth | argument and artifact archetype libraries, `Cartesian`; claim accuracy, domain scorers, `LLMGroundTruth` |
| Study one step of RL | `Seeds` + `bon_curve` (KL axis); `GameTree` + `tree_mesh` for proposal → critique → rebuttal |
| Go beyond the local policy | `PromptOptimizer` with LLM (OPRO / GEPA-style) or parameter proposers, steering, hidden GT constraints, held-out evaluation; `AgenticOptimizer` |
| Handle several optimisers at once | `PSRO` (level-k / fictitious play / double oracle) with opponent-aware LLM oracles; `EmpiricalGame` equilibria, regret, collusion |
| Run actual RL | `StrategyGradient` (REINFORCE or natural PG); `MechanismEnv` for any RL stack; TRL GRPO `reward_function`; DPO `preference_pairs` |
| Use real capability gaps | `ChessMoves`, `PrivateSQL`, `HiddenTestsCode`, `MiniF2F`, `ManifoldForecasting`, `ControlArenaDomain` |
| Vary verification systematically | `EvidencePolicy` (verifiers, budgets, noise, visibility, requests) + domain verifier strength + `sweep` |
| Monitor agents (actions, CoT, probes) and pay them | `Monitoring` + `MonitoredReward`, `SimulatedProbe` (fixed AUROC) vs `FunctionProbe` (real), audits, trainable monitors |
| Compare shared rewards with snitching incentives in swarms | `Swarm` with `TeamReward` vs `Whistleblower` on `AbstractSwarm` / `SimOps` (LLM tools); `theory.swarm_game` |
| Care about more than IC | `diagnostics` (accuracy, calibration, signal-to-noise, biases, compliance, cost), `coalition_gain` ([beyond_ic](docs/beyond_ic.md)) |
| Publish results before ground truth exists | `release.create_release` → `resolve_release` (Merkle-committed, static page) |
| Connect to mechanism design | [docs/theory.md](docs/theory.md) |

## Demos

| | |
|---|---|
| ![](docs/figures/monitoring_misbehaviour.png) | ![](docs/figures/swarm_basin.png) |
| Penalising a real monitor's alarms buys obfuscation; a fixed-accuracy probe deters | Swarms: the basin of the honest outcome, simulation vs mean-field theory |
| ![](docs/figures/hiddenbits_asd_budget.png) | ![](docs/figures/bon_honesty.png) |
| Debate's ASD grows with the verification budget | Best-of-N moves a liar's lies beyond the verification budget |

All seven: [docs/demos.md](docs/demos.md) · HTML gallery: [docs/gallery/index.html](docs/gallery/index.html).

## Documentation

- [Concepts and design](docs/concepts.md): mechanisms, domains and the firewall, verification,
  behaviours, agents, records, analyses, releases, extending.
- [Cookbook](docs/cookbook.md): recipes for every experiment type above.
- [Theory](docs/theory.md): scalable oversight as mechanism design.
- [Beyond incentive compatibility](docs/beyond_ic.md): what else to measure, and how.
- [ControlArena](docs/controlarena.md): relationship, what ControlArena's scores are, the bridges.
- [API reference](docs/reference.md): every public class and function, from docstrings.
- [Examples](examples/README.md) and [configs](configs/).

## Layout

```
src/oversight_arena/
  core/          tasks & GT firewall, roles, strategies/profiles, transcripts & evidence, rewards, records
  mechanisms/    debate, judging, critique, preference, monitoring, peer prediction, forecasting, swarm
  domains/       hidden bits, GSM8K, QuALITY, MCQ, code, SQL, chess, Lean, forecasting, swarms, monitoring
  channels/      verified claims (evidence policies, verifiers), GT channels, probes
  agents/ models/ sim/    LLM agents (Inspect), parsing, caching; programmatic agents
  experiment/    experiments, profile sources (Stances, Seeds, GameTree, Cartesian...), results, sweeps
  ground_truth/  generic, domain and model-based GT scorers
  elicitation/   strategy libraries, prompt optimisation, PSRO, RL adapters
  analysis/      IC metrics, best-of-n, games & dynamics, diagnostics, plots, HTML reports
  release/       commitments, releases, resolution
  theory/        disclosure, swarm game, monitoring (analytic companions)
  integrations/  Inspect, ControlArena
```

## Status

- **Tests.** The test suite covers the core, every mechanism, the analyses, the information
  firewall, releases, the theory modules and both integrations (ControlArena tests run when
  `control-arena` is installed; they were validated against 19.0.0).
- **Results so far.** The demos use programmatic agents and exact judges. They validate the
  machinery and the theory. Results with frontier LLMs are what the examples and configs are for.
- **Sandboxing.** Code from models runs confined (Landlock, seccomp and rlimits on Linux, plus a
  Python audit hook): no reads outside the Python installation, no network, no new processes. It
  fails closed where Landlock is unavailable (override with `OA_ALLOW_NO_KERNEL_SANDBOX=1` inside a
  container). Lean runs confined too. Grading runs trusted tests in a separate process, so graded
  code cannot fake its result (see the [cookbook](docs/cookbook.md#8-real-capability-gaps)). For adversarial workloads at scale, also
  use a container or plug in an executor (`domains._exec.set_executor`).

## Citing

If you use the ASD metric, cite Pallavi Sudhir, Kaunismaa & Panickssery (2025), *A Benchmark for
Scalable Oversight Protocols*. For best-of-n game trees, cite Gould et al. (2026), *Debate with
Self-Play Best-of-N Optimization*. Further references are in [docs/theory.md](docs/theory.md).
