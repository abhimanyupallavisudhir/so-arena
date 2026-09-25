# OversightArena

Experiments on **what oversight mechanisms incentivize**, including how those incentives change under optimization.

A mechanism is an arbitrary async workflow plus explicit rewards for its trainable participants. A judge can be trainable; an unscored role is a fixed fixture. Independent evaluators measure realized behavior after the mechanism finishes. Their labels never enter the mechanism runner or its reward-only optimizers.

This is an installable research library, with a dependency-free Python 3.11+ core, optional Inspect integration, and deterministic examples. It is a new implementation rather than a fork of ControlArena or SOlib.

## Run it

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,plots]'
oversight-arena demo --plots --output outputs/demo
python -m http.server 8000 --directory outputs/demo
```

Open `http://localhost:8000/report.html` and `http://localhost:8000/optimization.png`. The demo executes 109 runs, compares paired SQL interventions, optimizes prompts, trains categorical swarm strategies with REINFORCE, exports training trajectories, and publishes an unresolved forecast. **Its policies are scripted; the graphs are demonstrations, not findings about LLMs.**

```sh
printf '{"forecast-1": true}\n' > outputs/demo/resolutions.json
oversight-arena resolve outputs/demo/records outputs/demo/resolutions.json \
  --role forecaster --version resolution-1
oversight-arena report outputs/demo/records --output outputs/demo/resolved.html
pytest -q
```

Resolution appends an evaluation revision. It cannot replace the original rewards, transcript, or pending evaluation.

## What is implemented

| Area | Working components |
| --- | --- |
| Workflows | Arbitrary async mechanisms; debate, interactive consultancy, monitoring, sealed swarm reports; explicit rewards for trained judges |
| Information boundaries | Role-visible events, private interventions, tool permissions, simultaneous rounds, fresh policy/environment factories, call/time budgets |
| Sampling | Natural policies, replay, balanced or weighted prompt strata, paired seeded conditions |
| Optimization | Exact finite-pool best-of-N with ties; conditional multi-agent game trees; reward-only prompt search with held-out evaluation; simultaneous/alternating joint prompt search |
| Incentives | Paired ASD with task-bootstrap intervals and manipulation checks; covariance, concordance, selection regret, lower tails, gap recovered, detection metrics |
| Games and learning | Mixed-policy unilateral regret, pure equilibria, coalition gain, coarse-correlated regret, cyclic best responses; actual tabular multi-agent REINFORCE |
| Domains | Private SQL execution suites; answer scorers; delayed forecasts; chess legality and fixed-node UCI engine scoring; callback evaluators for additional domains |
| Evidence | Content-bound claims, versioned verifiers, checker-defined scope; exact/proxy/human/outcome provenance |
| Integrations | Inspect model policies, LLM prompt proposer, Inspect tasks, explicit ControlArena/Inspect log mappings, reward trajectory exports |
| Artifacts | Immutable runs, versioned evaluation sidecars, mechanism-only JSONL, HTML reports, blinded human-review packets, matplotlib plots and CSV |

External LLM training systems can consume the reward exports; this release does not implement neural PPO/GRPO or a hosted training service. Code/Lean execution, database/git/payment snapshots, and human participant recruitment belong to supplied environment/evaluator adapters. The SQL suite is executable locally; a full realistic long-horizon firm benchmark is not bundled. See [scope and limitations](docs/design.md#scope-and-limitations).

## Define a mechanism

```python
import asyncio
from oversight_arena import Action, Outcome, Reward, Role, Task, run

async def worker(observation):
    return Action("A proposed answer", {"answer": 42})

async def judge(observation):
    # Replace with InspectPolicy("provider/model") for real model calls.
    return Action(data={"endorsement": 0.8})

async def mechanism(ctx):
    await ctx.act("worker", "Answer the task.")
    verdict = await ctx.act("judge", "Return data.endorsement in [0,1].")
    return Outcome({"worker": Reward(verdict.data["endorsement"])})

record = asyncio.run(run(
    Task("example", "What is the answer?"),
    (Role("worker"), Role("judge", trainable=False)),
    {"worker": worker, "judge": judge}, mechanism,
    name="direct-review", manifest={"policies": "example-v1", "reward_rule": "endorsement"},
))
assert record.reward("worker") == 0.8
```

`AnswerScorer`, `SQLScorer`, `ForecastScorer`, or a custom evaluator can now evaluate `record`. A mechanism reward and an independent quality score may disagree: that disagreement is the object of study.

## Read next

- [Design and extension contracts](docs/design.md): information boundaries, trainable judges, tools, domains, reproducibility.
- [Experiment guide](docs/experiments.md): paired ASD, prompt strata, multi-agent optimization, delayed labels, real models.
- [Mechanism design and research methodology](docs/theory.md): estimands, equilibrium assumptions, swarm incentives, and concerns beyond incentive compatibility.
- [Review of the lab PDF](docs/lab-review.md): relevant recommendations, corrections, and what this implementation covers.
- [Integration guide](docs/integrations.md): Inspect, ControlArena, external trainers, human ratings, versioned source references.
- [Executable examples](examples/): use the library with scripts and optional real model backends.

The core distinction follows the [ASD paper](https://arxiv.org/abs/2504.03731). Finite-tree optimization is informed by the [self-play best-of-N study](https://www.lesswrong.com/posts/hb8pv3zyAHGJpwz9F/debate-with-self-play-best-of-n-optimization). These methods are stress tests under specified behavior support, not proofs of robustness to arbitrary future training.
