"""Dry-run models for estimating the cost of an experiment before running it.

Following the approach of ``costly``: run the *whole* experiment with every real model replaced by a
:class:`SimulatedModel` that returns filler text of a plausible length and records the usage the
real model would have incurred (priced from the model registry). Because transcripts grow as they
would in a real run, input-token counts for later turns are realistic. Downstream parsers fall back
gracefully (uniform probabilities, first option) on filler text, so every protocol branch runs.

Usage::

    so_arena.configure(simulate=True)
    result = experiment.run()          # no API calls
    print(result.usage_summary())      # tokens and $ by model and role
"""

from __future__ import annotations

import hashlib
import random

from so_arena.core.types import Completion, Usage
from so_arena.models.base import Model, messages_tokens
from so_arena.models.registry import get_spec

_WORDS = (
    "the argument evidence shows that answer claim supports because therefore we see clearly "
    "consider case however indeed first second finally which suggests strong reason data"
).split()


class SimulatedModel(Model):
    def __init__(self, name: str, *, output_tokens: int = 200, output_tokens_sd: float = 0.3, seed: int = 0):
        self.name = name
        self.spec = get_spec(name)
        self.mean_output_tokens = output_tokens
        self.output_tokens_sd = output_tokens_sd
        self.seed = seed
        self.supports_logprobs = bool(self.spec and self.spec.supports_logprobs)

    async def generate(self, messages, options=None, *, sample_index=0):
        key = f"{self.seed}|{sample_index}|{len(messages)}|{messages[-1].content[:256] if messages else ''}"
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
        max_tokens = options.max_tokens if options and options.max_tokens else None
        n = max(1, int(rng.lognormvariate(0, self.output_tokens_sd) * self.mean_output_tokens))
        if max_tokens:
            n = min(n, max_tokens)
        # ~0.75 words per token
        text = " ".join(rng.choice(_WORDS) for _ in range(max(1, int(n * 0.75))))
        usage = Usage(input_tokens=messages_tokens(messages), output_tokens=n, calls=1)
        if self.spec is not None:
            usage = usage.model_copy(update={"cost_usd": self.spec.cost(usage) or 0.0})
        return Completion(text=text, model=f"sim/{self.name}", usage=usage, metadata={"simulated": True})
