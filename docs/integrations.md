# Integrations and reproducibility

## Inspect models

```python
from oversight_arena.adapters.inspect import InspectPolicy

worker = InspectPolicy("provider/worker-model", json_output=True,
                       generation={"temperature": 0.7, "max_tokens": 2048})
judge = InspectPolicy("provider/judge-model", json_output=True)
```

Install `.[inspect]` and use the provider's ordinary Inspect configuration. The adapter sends the already filtered observation, requests JSON when selected, uses the call seed and preserves token usage. A pricing callback may compute cost from the returned Inspect `ModelOutput`; otherwise cost is explicitly unknown. Provider-side retries and nondeterminism remain provider behavior. Malformed JSON fails the run rather than silently assigning a low reward.

`InspectPromptProposer` wraps a plain-text Inspect policy for prompt search. It receives `SearchRequest` containing public mechanism description, peer prompts and scalar reward history. Neither model adapter directly receives an oracle. Standard Inspect model interfaces are documented in the [official API reference](https://inspect.aisi.org.uk/reference/inspect_ai.model.html).

## ControlArena bridge

The concrete, tested example is [controlarena_bridge.py](../examples/controlarena_bridge.py). It uses public ControlArena APIs to construct a setting/micro-protocol task, adds `mechanism_reward_scorer`, runs it through Inspect and imports the resulting `EvalSample`.

For an existing experiment:

1. Define the actual training payoff for each trainable participant. For example, a worker might receive a specified task reward minus an audit penalty. A raw suspicion score is a signal; whether and how it becomes a reward is a design choice.
2. Add each `mechanism_reward_scorer(role, async_reward_callback)` to the `monitoring_scorers` passed to `get_control_task`. It receives `TaskState`, not the hidden `Target`. A callback must still avoid privileged information in task metadata or closures.
3. Pass explicit `reward_map` and optional `truth_map` selectors to `from_eval_sample` or `read_controlarena_log`. Categorical values like `C`/`I` require an explicit conversion map. Reward and truth sources must be distinct. All trainable roles must receive reward mappings.
4. Supply a sanitized `PublicTask` and, if useful, an event-projection function. Raw Inspect inputs, targets and metadata are intentionally not copied; they can contain attack instructions or oracle data. Assign transcript authors using your protocol's real role attribution rather than guessing from `assistant` messages.

The bridge retains sample id/epoch and mapping provenance. It does not infer ControlArena's multi-agent credit assignment, translate an arbitrary micro-protocol into OversightArena's `Context`, or automatically turn task accuracy into a training reward. It lets the two execution frameworks share the same incentive-analysis layer. [ControlArena micro-protocols](https://control-arena.aisi.org.uk/micro_protocols.html), [task construction API](https://control-arena.aisi.org.uk/reference/eval.html).

The optional integration was exercised with ControlArena 19.0.0 and Inspect AI 0.3.268. Core installation does not install either dependency.

## Delayed labels and public release

```bash
oversight-arena report runs/demo/runs.sqlite --output runs/demo/report.html
oversight-arena export runs/demo/runs.sqlite --output runs/demo/public.json
oversight-arena resolve-forecast runs/demo/runs.sqlite EPISODE_ID \
  --role forecaster --outcome 1 --source 'Resolution source URL or identifier' \
  --resolved-at '2026-09-25T00:00:00Z'
```

Re-render after resolution. This appends an evaluation revision without changing the original event/reward hash. Corrections can append another revision; previous labels remain available. The CLI prints the affected episode id. Unknown or pending results remain visibly unresolved in reports.

For any other domain, load an episode with `RunStore.get`, call `evaluate(task, episode, scorer)`, and save the returned record. Use a fresh scorer version when changing evaluation semantics. The public projection exposes public messages, final outputs, scalar rewards and evaluation values, omitting private messages/channels and evaluator evidence. It is a local export; the package does not publish to a hosting service automatically.

## Reproducible runs

Record the mechanism name/configuration, reward rule version and normalization, role trainability, model/weight versions, prompt hashes or text, generation settings, dataset snapshot/splits, tool and evaluator versions, sampling seeds, update schedule, search budget and policy population ordering. Use `provenance` for these identifiers. Provider credentials should remain in the environment, never in prompts or provenance.

Content hashes ensure stored episodes can be checked, but they cannot make external APIs deterministic or prove the origin of an untrusted database. Keep original Inspect logs for model-backed runs and exact environment locks with reported experiments. Bootstrap units should reflect actual independent tasks/task families. Describe unresolved-label and failed-run denominators; successful-only analysis can exaggerate a protocol's performance.

`requirements-tested.txt` records the complete Python 3.11 optional-integration environment used for verification. Recreate it with `pip install -r requirements-tested.txt` then `pip install --no-deps -e .`; it is not the minimal runtime dependency list.
