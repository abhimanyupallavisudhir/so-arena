"""PeerPrediction reporters get byte-identical prompts (system prompt built for reporter_1, same
question, same private context) and the same sample_index, and the response-cache key has no role:
once the cache is warm every reporter receives the SAME cached completion, so 'independent'
reports agree perfectly (output agreement pays everyone 1; BTS/multitask see no disagreement)."""
import random, tempfile
from so_arena import configure
from so_arena.core.types import Completion
from so_arena.core.items import TaskItem, AnswerOption
from so_arena.core.policy import LLMPolicy
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.mechanisms.elicitation import PeerPrediction
from so_arena.models import inspect_backend

rng = random.Random(0)
calls = {"n": 0}
async def fake_generate(self, messages, options=None, *, sample_index=0):  # a temperature-0.7 sampler
    calls["n"] += 1
    import asyncio; await asyncio.sleep(0.01)  # network latency
    return Completion(text=f"Answer: {rng.choice('AB')}", model=self.name)
inspect_backend.InspectModel.generate = fake_generate
configure(cache_dir=tempfile.mkdtemp())

items = [TaskItem(id=f"q{i}", question=f"Question {i}?", answers=[AnswerOption(label="A", text="yes"), AnswerOption(label="B", text="no")])
         for i in range(6)]
mech = PeerPrediction(n_reporters=4, rule="output_agreement")
prof = Profile(name="llm", players={f"reporter_{k+1}": LLMPolicy("openai/some-model") for k in range(4)})
for run in ("first run (cold cache)", "second run (warm cache)"):
    calls["n"] = 0
    eps = run_sync(run_episodes(mech, items, [prof], ground_truth=[]))
    agree = sum(len(set(e.outcome.data["answers"].values())) == 1 for e in eps)
    mean_r = sum(sum(e.rewards.values()) / 4 for e in eps) / len(eps)
    print(f"{run}: API calls={calls['n']}, items with unanimous reports={agree}/{len(eps)}, mean reward={mean_r:.2f}")
