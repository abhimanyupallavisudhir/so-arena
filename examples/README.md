# Examples

Every script runs out of the box. Scripts with LLM agents read their models from
`OA_EXPERT_MODEL` / `OA_JUDGE_MODEL` (any Inspect model name, e.g. `openai/gpt-4.1`,
`anthropic/claude-sonnet-4-5`, `openrouter/meta-llama/llama-3.1-8b-instruct`). Without them they
fall back to Inspect's `mockllm/model`, which checks the pipeline but produces meaningless numbers.
Model calls are cached in `.oa_cache/`, so re-runs are free.

| Script | What it shows | Needs |
|---|---|---|
| `verification_sweep.py` | How verification budget and judge credulity shape IC across protocols; exact Bayesian judges | nothing |
| `chess_llm_judge.py` | Real capability gap: engine-backed experts, an LLM (or engine) judge, verified lines | Stockfish |
| `asd_llm.py` | Agent Score Difference for NaiveJudge / Propaganda / Consultancy / Debate with LLMs | models |
| `best_of_n.py` | One step of optimisation: best-of-n pools, reward-vs-GT curves with a KL axis; game trees | models |
| `prompt_search.py` | Steered prompt optimisation: honest-only vs deceptive-only frontiers, held-out evaluation | models |
| `psro_debate.py` | Multi-agent prompt optimisation (PSRO / level-k) with opponent-aware LLM optimisers | models |
| `swarm_llm.py` | LLM workers in SimOps: shared reward vs whistleblower bounties | models |
| `monitoring_obfuscation.py` | Optimising an actor against a CoT monitor vs a fixed-AUROC probe (Proposition 4) | models |
| `forecast_release.py` | Judged forecasts on open Manifold questions: release now, resolve later | models, network |
| `inspect_task.py` | The same experiments as Inspect tasks (`inspect eval examples/inspect_task.py`) | models |
| `control_arena_bridge.py` | ControlArena logs as reward-rule IC analyses; OversightArena mechanisms as micro-protocols | `control-arena` |
| `rl_trl_grpo.py` | Weight-level RL with TRL GRPO using a mechanism's reward (sketch; needs a GPU) | `trl` |

Configuration-file versions of these experiments are in `../configs/oversight_arena/`
(`oversight-arena run configs/oversight_arena/<file>.yaml`).
