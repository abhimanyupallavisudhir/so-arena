"""The response cache key is (model name, messages, options, sample_index): a provider model's
model_args (base_url, model_path, adapter, ...) are not part of it, so two different backends
served under one name (e.g. RL checkpoints behind vllm/policy) share cached completions."""
import tempfile
from so_arena import configure
from so_arena.core.types import Completion, Message, GenerateOptions
from so_arena.models import inspect_backend
from so_arena.models.base import get_model
from so_arena.core.runner import run_sync

async def fake_generate(self, messages, options=None, *, sample_index=0):  # stands in for the API call
    return Completion(text=f"answer from {self._model_args['base_url']}", model=self.name)
inspect_backend.InspectModel.generate = fake_generate

configure(cache_dir=tempfile.mkdtemp())
ckpt0 = get_model("vllm/policy", base_url="http://step0")
ckpt9 = get_model("vllm/policy", base_url="http://step900")
print("distinct model objects:", ckpt0 is not ckpt9)
msgs = [Message.user("Is the proof correct?")]
opts = GenerateOptions(temperature=0.7)
print("step0  :", run_sync(ckpt0.generate(msgs, opts)).text)
out = run_sync(ckpt9.generate(msgs, opts))
print("step900:", out.text, "| cached:", out.cached)

# the same blind spot in episode ids: LLMPolicy.describe() has the model name only
from so_arena.core.policy import LLMPolicy
from so_arena.core.runner import Profile, profile_description
p0 = Profile(name="rl", players={"agent": LLMPolicy(ckpt0)})
p9 = Profile(name="rl", players={"agent": LLMPolicy(ckpt9)})
print("profile descriptions equal (-> same episode id, resume reuses step-0 episodes):",
      profile_description(p0) == profile_description(p9))
