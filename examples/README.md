# Examples

From the repository root, after `pip install -e .`:

| Command | Demonstrates |
|---|---|
| `oversight-arena demo --output runs/demo` | SQL scoring, paired interventions, BoN, prompt search, REINFORCE, a reporting game, delayed labels, persistence and reports |
| `oversight-arena demo --output runs/demo --plots` | Also creates an exportable reward-versus-quality SVG; requires `.[plots]` |
| `python examples/prompt_search.py` | Two-agent simultaneous and alternating updates and independent honest/deceptive/unrestricted search cohorts |
| `python examples/verified_artifact.py` | Bounded agent tool use, public artifact verification and independent private specification checking |
| `python examples/controlarena_bridge.py` | Actual ControlArena task, Inspect log, explicit reward/truth import; requires `.[controlarena]` |

Every included example is offline and uses fixtures. To make a model-backed experiment, replace fixture policies with `InspectPolicy` or another async policy, specify generation/model versions and prices, and use a versioned task dataset. Nothing in the examples establishes that a real oversight mechanism is robust to model training.
