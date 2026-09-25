# Integration guide

## Inspect and ControlArena

ControlArena's microprotocols coordinate policies and monitors; its settings supply tasks, tools and environments. Its scorer outputs include task outcomes and suspicion measures. These are not automatically an individual training reward. OversightArena adds an explicit reward contract and independent behavioral evaluation. See the primary [microprotocol reference](https://control-arena.aisi.org.uk/reference/micro_protocol.html), [settings reference](https://control-arena.aisi.org.uk/reference/settings.html), and [scorer guide](https://control-arena.aisi.org.uk/scorers.html).

The boundary is deliberate: a mechanism is reusable across domains, while settings and evaluators adapt domain-specific observations and ground truth. Some protocols will still need domain-specific verification tools or question formats; these are explicit dependencies rather than hidden assumptions.

```sh
pip install -e '.[inspect]'
python examples/llm_sql.py --worker provider/model --judge provider/model --output outputs/llm-sql
```

Use existing provider environment variables. The example makes paid calls if the specified backend does; it is not run automatically by installation, tests or the demo. `InspectPolicy` uses Inspect's `get_model`, messages and `GenerateConfig`, with strict JSON `Action` parsing. Invalid output fails the run instead of becoming a zero reward. Generation parameters, retries and provider limits are set in `config`. No plaintext API keys are written to records. Model provider costs and semantics remain those of Inspect's [model API](https://inspect.aisi.org.uk/models.html).

`as_inspect_task(tasks, experiment, condition)` returns an actual Inspect task. The solver stores a serializable `oversight_run` in sample metadata; the scorer uses `reward/<role>` keys. It does not copy independent references into Inspect targets. A test runs this with Inspect's mock backend, writes a real `.eval` log, then reads it with `load_control_log`.

Import an existing ControlArena run:

```python
from oversight_arena.adapters import import_control_sample, load_control_log

# for sample in load_control_log("logs/control.eval"):
#     record, evaluation = import_control_sample(
#         sample,
#         source="sha256:hash-of-this-log",
#         snapshot="repository-sha:database-revision",  # common across paired arms
#         role="worker",
#         reward_scorer="suspicion",
#         reward_transform=lambda value: 1 - float(value),
#         quality_scorers={"task_success": ("quality", "exact")},
#         quality_transforms={"task_success": lambda value: float(value == "C")},
#         condition="deceptive",
#         mapping_version="study-v1",
#     )
```

Replace scorer names and mappings with those actually in the log. The example reward transform assumes normalized suspicion in $[0,1]$; normalizing a different range must be explicit. A task-success scorer implemented by another LLM should be labelled `proxy`, not `exact`. Honest/attack mode labels describe experimental conditions, not audited behavior. `source` preserves log provenance; `snapshot` preserves the common starting state needed for paired comparisons. Increment `mapping_version` when changing reward or quality interpretation.

Imported messages are private by default because Inspect transcripts can contain system or attack instructions. Raw sample input is also excluded from the public task; pass an explicitly reviewed `public_prompt` to override its neutral placeholder. Build a domain-specific public projection before sharing them. Post-hoc import cannot recover unlogged per-agent observations, conditional continuations, hypothetical incentives or missing reward components. A log-derived reward rule is a proposed training mechanism; agents in the original control run were not necessarily trained to optimize it.

Compatibility evidence: optional dependencies were installed and tested against **Inspect 0.3.268** and **python-chess 1.999 / chess 1.11.2**. ControlArena documentation was reviewed on 2026-09-25. No live cloud ControlArena environment or paid LLM training was run. The core remains usable without either package; CI tests the core separately.

## External training and prompt search

`training_rows(records)` exports full private observations, emitted actions, role ID and terminal mechanism reward for every trainable role in a successful training-split run. Keep these records private: they include assigned instructions. Store the policy checkpoint, tokenizer, optimizer state, opponent population and update schedule alongside them. A neural RL backend must implement tokenization, log-probabilities, advantages, off-policy checks, masking, KL costs and its chosen credit assignment. A fixture must remain frozen in that backend too.

`prompt_search` and `joint_prompt_search` are backend-independent. An optimizer can call DSPy, GEPA or another search service inside `propose`, but the interface alone is not a tested vendor-specific integration. `InspectPromptProposer` provides one concrete LLM-based optimizer. No dependency on an optimizer's internal scoring convention is imposed.

## Human and verifier adapters

`review_packet` creates a packet with public task text, public evidence, rubric and an opaque packet identity. It omits mechanism reward, condition metadata and private instructions. `import_rating` creates a human-provenance sidecar containing rater identity and review time. Multiple raters should use distinct scorer IDs; prespecify aggregation and adjudication before looking at outcomes. A hosted rating queue, participant consent and time enforcement are outside this release.

`VerifierRegistry` accepts async checkers returning `(True | False | None, detail)`. Register an explicit version and scope. It can be exposed as a role-permitted tool. Errors become verification errors, not false claims; an unavailable checker returns unknown. Record any checking costs and disclosure choices as part of the mechanism. See [verified claims example](../examples/verified_claims.py).

## Sources and provenance

Primary technical sources reviewed for the design:

- [A Benchmark for Scalable Oversight Protocols, arXiv:2504.03731v1](https://arxiv.org/html/2504.03731v1): separate agent incentives from judge accuracy, and compare counterfactual rewards.
- [Debate with Self-Play Best-of-N Optimization](https://www.lesswrong.com/posts/hb8pv3zyAHGJpwz9F/debate-with-self-play-best-of-n-optimization): conditional tree optimization as an inference-time stress test.
- [ControlArena](https://control-arena.aisi.org.uk/): existing control environment and workflow infrastructure.
- [Inspect evaluation logs](https://inspect.aisi.org.uk/reference/inspect_ai.log.html): typed run/log interoperability.
- User-supplied *Oversight That Scales*, 17 September 2026, 84 pages: [review](lab-review.md).

No source repository code or benchmark datasets were copied. The MIT license covers this original implementation; it does not relicense SOlib, the supplied PDF, or third-party datasets. A source manifest is stored in [research/sources.json](../research/sources.json).
