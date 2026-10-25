# Attempt 4: (a) agents observe task.split (train vs test); (b) InspectPolicy publishes _model/_usage
import asyncio, json, sys, types
sys.path.insert(0, "/tmp/att4/src")

# --- minimal stub of inspect_ai.model so InspectPolicy can run offline ---
m = types.ModuleType("inspect_ai.model")
class _Msg:
    def __init__(self, content): self.content = content
m.ChatMessageSystem = m.ChatMessageUser = _Msg
m.GenerateConfig = lambda **k: k
class _Usage:
    def __init__(self, n): self.n = n
    def model_dump(self): return {"output_tokens": self.n}
class _Out:
    def __init__(self, c, n): self.completion, self.usage = c, _Usage(n)
class _Model:
    def __init__(self, name): self.name = name
    async def generate(self, msgs, config=None):
        obs = json.loads(msgs[1].content)
        if obs["role"] == "judge":
            # judge just echoes what it could see about the worker
            w = [e for e in obs["events"] if e["actor"] == "worker" and e["kind"] == "action"][-1]
            return _Out(json.dumps({"data": {"scores": {"worker": 0.5},
                                             "saw": {k: v for k, v in w["data"]["data"].items() if k.startswith("_")},
                                             "saw_split": obs["task"]["split"]}}), 5)
        return _Out(json.dumps({"text": "answer", "data": {"answer": 1}}), 900)
m.get_model = lambda name: _Model(name)
pkg = types.ModuleType("inspect_ai"); pkg.model = m
sys.modules["inspect_ai"] = pkg; sys.modules["inspect_ai.model"] = m

from oversight_arena.adapters import InspectPolicy
from oversight_arena.mechanisms import Consultancy
from oversight_arena.runtime import run
from oversight_arena.types import Role, Task

async def main():
    for split in ("train", "test"):
        r = await run(Task("t1", "Q?", split=split), (Role("worker"), Role("judge", False)),
                      {"worker": InspectPolicy("strong-provider/big-model"),
                       "judge": InspectPolicy("weak-provider/small-model")},
                      Consultancy("worker", rounds=1), name="c")
        print(split, r.status, "judge output:", json.dumps(r.outcome.output["judgment"]))
asyncio.run(main())
