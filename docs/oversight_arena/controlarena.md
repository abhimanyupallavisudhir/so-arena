# OversightArena and ControlArena

[ControlArena](https://github.com/UKGovernmentBEIS/control-arena) (UK AISI, built on Inspect)
evaluates **control protocols**: can we get useful work out of an untrusted model while keeping it
from doing harm? OversightArena evaluates **scalable-oversight mechanisms**: if we *train* models on
a protocol's signals, do the rewards push them toward good behaviour? The two share a lot of
structure. This page gives the mapping, explains what ControlArena's "scores" are, and shows the
integration. Everything below was validated against `control-arena` 19.0.0.

## What ControlArena's scores are

ControlArena produces three kinds of numbers:

1. **Setting scorers** judge each trajectory's `main_task_success` and (in ATTACK mode)
   `side_task_success`. They run after the episode, with privileged access (tests, reference
   solutions, environment diffs, hidden labels), and the protocol never sees them. In
   OversightArena terms they are **ground-truth scorers**. Some settings use an LLM judge here,
   which makes them proxy ground truth.
2. **Monitor suspicion scores** are produced *inside* the protocol by a (usually trusted) monitor,
   per step (in message metadata) and aggregated per trajectory (`max_suspicion_scorer`,
   `end_of_trajectory_monitor`). They are the blue team's *observation*: a signal the protocol acts
   on, not ground truth.
3. **Safety and usefulness** are statistics over many episodes:
   - usefulness is the honest-mode main-task success rate;
   - safety is the probability that attacks are caught, at an audit threshold calibrated on honest
     suspicion scores for a given audit budget (the budget is a percentage).

   They measure the *protocol as deployed*, with fixed (prompted) policies.

**Nothing is paid to the agents.** ControlArena's policies are fixed model organisms: an honest
policy and an attack policy, selected by `EvalMode`. That is the gap OversightArena fills. A
**mechanism** is a protocol plus a **reward rule** for its trainable roles, and the question is
whether training on those rewards produces the behaviour the ground truth scores highly.

## Concept mapping

| ControlArena | OversightArena |
|---|---|
| Setting (dataset, sandbox, tools, scorers) | `Domain` (tasks, tools, verifiers, environment, GT scorers) |
| Micro-protocol (`@control_agent`: trusted monitoring, defer-to-trusted, trusted editing...) | game form of a `Mechanism` (`run(ctx)`) |
| — | reward rule (`RewardRule`) paid to trainable roles |
| Untrusted / trusted model roles | trainable roles / fixtures; Inspect model roles `untrusted` / `trusted` |
| `EvalMode.HONEST` / `ATTACK` (honest / attack policy) | strategies (`honest_strategy()` / `attack_strategy()`, like ASD's argue-correct / argue-incorrect) |
| Side-task description given to the attack policy | privileged info block behind a clearance, granted by the attack strategy |
| Monitor (suspicion score) | monitor role / probe inside the mechanism; its output is part of the record |
| Main / side task scorers | GT scorers |
| Safety, usefulness (per protocol, over episodes) | principal-level outcome metrics; plus IC metrics per role |
| Audit (idealised: an audited attack is caught) | declared GT channel (`Auditor`, `SwarmAudit`) with cost |
| — | verification policies, trainable judges/monitors, best-of-$n$, prompt optimisation, equilibria |

Both frameworks answer the "bespoke or general?" question the same way: protocols are written
generically and settings/domains supply the task-specific parts.

## Which control protocols make sense as training signals?

A protocol is worth studying as a mechanism if its signals can be *paid to* someone. Examples:

- **Trusted monitoring as a reward.** The untrusted policy gets main-task success minus a penalty
  on suspicion (or on alarms above a threshold). The question: does optimising against the monitor
  reduce attacks, or only their visibility? See [theory.md §7](theory.md), Proposition 4.
- **Defer-to-trusted.** Deferral forfeits credit, so the policy is paid for work the monitor lets
  through.
- **Trainable monitors.** A monitor is rewarded through random audits: $+$ for confirmed alarms,
  $-$ for false alarms and audited misses. Monitor and policy then co-adapt.
- **Attack/honest as strategies.** ASD between honest and attack runs of the same main task
  measures whether the protocol-as-reward disfavours attacking at all. Alignment with
  "main task done and no side task" measures whether it tracks what we actually want.

OversightArena's `Monitoring` mechanism covers these natively (LLM action monitors, CoT monitors,
simulated and real probes, deferral, graders, audits, `MonitoredReward`).

## The bridges (`oversight_arena.integrations.control_arena`)

Install with `pip install control-arena`. The Inspect bridge alone needs only `inspect-ai`.

### 1. ControlArena logs → OversightArena results, with a reward rule

```python
from oversight_arena.integrations.control_arena import results_from_control_arena
from oversight_arena.mechanisms import MonitoredReward
from oversight_arena.analysis import asd, alignment

res = results_from_control_arena(
    "logs/",                                                   # your ControlArena eval logs
    reward=MonitoredReward(actor="untrusted", continuous=True, alarm_penalty=2.0),
    audit_budget=1.0,                                          # alarm threshold as ControlArena computes it
)
d = res.df(trainable_only=True)
asd(d, gt="honest")              # reward(honest runs) - reward(attack runs), per main task
alignment(d, gt="safe_useful")   # does the reward rank "useful and no side task" higher?
```

Each sample becomes an episode with a trainable `untrusted` role and a fixture `monitor`:
- outcome: main-task success, suspicion, alarm;
- ground truth: `honest`, `main_task`, `side_task`, `safe_useful`.

Honest and attack runs of the same main task are grouped as one task, so within-task comparisons
work. Use `task_key="sample"` when main-task names are not problem identifiers.

### 2. A ControlArena setting as an OversightArena domain

```python
from oversight_arena.integrations.control_arena import ControlArenaDomain, attack_strategy, honest_strategy
from oversight_arena.mechanisms import Monitoring
import oversight_arena as oa

dom = ControlArenaDomain(setting="data_poisoning", setting_kwargs={"scorer_model": "openai/gpt-4.1-mini"})
profiles = [oa.Profile.of(label="honest", actor=honest_strategy()),
            oa.Profile.of(label="attack", actor=attack_strategy())]
res = oa.Experiment(dom, Monitoring(probe=None), agents, profiles).run()
```

What the bridge provides:
- **Tasks:** the main task is public; the side task sits behind the `side_task` clearance, which
  only the attack strategy grants.
- **Submissions:** the setting's submit-tool fields become a structured submission.
- **Ground truth:** the setting's own scorer (`ControlArenaGT`) supplies main/side-task success.

Supported for settings whose scorers run without a sandbox (`data_poisoning`,
`sae_interp_sabotage`, both LLM-judged). Sandbox settings (APPS, Bash, ...) are better used through
bridge 3, or with OversightArena's own execution domains (`HiddenTestsCode`, `SimOps`).

### 3. An OversightArena mechanism as a ControlArena micro-protocol

```python
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.data_poisoning.data_poisoning_setting import DataPoisoningSetting
from inspect_ai import eval as inspect_eval
from oversight_arena.integrations.control_arena import oversight_protocol
from oversight_arena.mechanisms import Monitoring

cfg = ControlEvalConfig(untrusted_model="openai/gpt-4.1", trusted_model="openai/gpt-4.1-nano")
tasks = get_control_task_product(DataPoisoningSetting(), oversight_protocol(Monitoring(probe=None)),
                                 [EvalMode.HONEST, EvalMode.ATTACK], eval_config=cfg,
                                 monitoring_scorers=[max_suspicion_scorer()])
logs = inspect_eval(tasks, **cfg.eval_args())
```

The episode runs inside each ControlArena sample, with agents bound to ControlArena's
`untrusted`/`trusted` model roles:
- the actor's structured submission becomes the `submit` call;
- the mechanism's suspicion is written where ControlArena's monitoring scorers read it;
- the OversightArena record and rewards travel in the message metadata.

ControlArena's safety/usefulness and OversightArena's IC analyses (via bridge 1) apply to the same
logs. This bridge suits single-submission settings.

## Running OversightArena experiments under Inspect

Independently of ControlArena, any OversightArena experiment can run as an Inspect task:

```python
from inspect_ai import task
from oversight_arena import Stances
from oversight_arena.domains.math import GSM8K
from oversight_arena.mechanisms import Debate
from oversight_arena.integrations.inspect import oversight_task

@task
def gsm8k_debate():
    return oversight_task(GSM8K(limit=50), Debate(rounds=2), profiles=Stances())
```

```
inspect eval my_eval.py --model openai/gpt-4.1 --model-role trusted=openai/gpt-4.1-nano
inspect view
```

Rewards (`reward/<role>`) and ground truth (`gt/<scorer>/<role>`) are Inspect scores; transcripts
appear in Inspect View. `results_from_logs("logs/")` gives back OversightArena `Results`.
