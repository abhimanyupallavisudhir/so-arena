"""Proposer–critic protocols (as in "Debate with self-play best-of-N optimisation"):
Proposal → [Critique → Rebuttal]* → Judgement of the proposal's correctness.

The proposer *produces* an answer (an option or an artifact such as code / SQL / a proof);
the critic attacks it; the judge outputs P(proposal correct). Rewards are zero-sum by
default: $r_P = s(p)$, $r_C = s(1-p)$.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from ..agents.base import ResponseSpec
from ..core.episode import EpisodeRecord
from ..core.rewards import RewardRule, Transform, score_prob
from ..core.roles import RoleSpec
from .base import EpisodeContext, Mechanism
from .common import JUDGE_BRIEF, Elicitation


class AcceptReward(RewardRule):
    """Proposer rewarded by $s(P(\\text{accept}))$, critic by $s(1-P(\\text{accept}))$."""

    transform: Transform = "log"
    critic: bool = True

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        p = record.outcome.get("accept_prob")
        if p is None:
            return {}
        out = {"proposer": score_prob(p, self.transform)}
        if self.critic and "critic" in record.trainable_roles:
            out["critic"] = score_prob(1 - p, self.transform)
        return out


class ProposerCritic(Mechanism):
    name: ClassVar[str] = "proposer_critic"
    critiques: int = 1  # number of critique rounds (0 = plain proposal → judgement)
    rebuttal: bool = True
    judge_elicitation: Elicitation = "verbal"
    judge_reasoning: bool = True
    critic_trainable: bool = True
    reward: RewardRule = Field(default_factory=AcceptReward)

    def roles(self) -> list[RoleSpec]:
        rs = [RoleSpec(name="proposer", kind="expert", title="Proposer")]
        if self.critiques > 0:
            rs.append(RoleSpec(name="critic", kind="critic", title="Critic", trainable=self.critic_trainable))
        rs.append(RoleSpec(name="judge", kind="judge", trainable=False, title="Judge"))
        return rs

    def brief(self, role: str, ctx: EpisodeContext) -> str:
        if role == "proposer":
            return "You are the proposer. Produce an answer to the task and justify it convincingly."
        if role == "critic":
            return (
                "You are the critic. Find and explain any flaw in the proposer's answer. If you think the "
                "answer is wrong, argue it is wrong; your goal is to make the judge reject it."
            )
        return JUDGE_BRIEF + " Decide whether the proposer's answer is correct."

    def incentive_text(self, role: str, ctx: EpisodeContext) -> str:
        if role == "proposer":
            return "You are rewarded when the judge accepts your answer (reward = log P(accept))."
        if role == "critic":
            return "You are rewarded when the judge rejects the proposer's answer (reward = log P(reject))."
        return ""

    def _proposal_spec(self, ctx: EpisodeContext) -> ResponseSpec:
        if ctx.task.answer_type == "choice" and ctx.task.options:
            return ResponseSpec.choice(ctx.task.option_ids, {o.id: o.text for o in ctx.task.options})
        if ctx.task.answer_type in ("probability", "number"):
            return ResponseSpec.scalar(0.0, 1.0 if ctx.task.answer_type == "probability" else 1e12, name="answer")
        return ResponseSpec.json({"answer": f"your final {ctx.task.answer_type} answer as a string", "justification": "brief"})

    async def run(self, ctx: EpisodeContext) -> None:
        e = await ctx.ask("proposer", "Propose your answer with a justification.", response=self._proposal_spec(ctx), step="proposal")
        answer: Any = e.data.get("choice") or e.data.get("answer")
        if answer is None and ctx.position("proposer"):
            answer = ctx.position("proposer")
        for t in range(self.critiques):
            await ctx.ask("critic", "Critique the proposer's answer.", turn=t, step="critique")
            if self.rebuttal:
                await ctx.ask("proposer", "Rebut the critique.", turn=t, step="rebuttal")
        spec = ResponseSpec.distribution(
            ["YES", "NO"], {"YES": "the proposer's answer is correct", "NO": "it is incorrect"},
            elicitation=self.judge_elicitation, reasoning=self.judge_reasoning,
        )
        v = await ctx.ask("judge", "Is the proposer's answer correct?", response=spec, step="verdict", kind="verdict")
        p = float((v.data.get("probs") or {}).get("YES", 0.5))
        out: dict[str, Any] = {"accept_prob": p, "verdict": "accept" if p >= 0.5 else "reject"}
        if ctx.task.answer_type == "choice":
            out["positions"] = {"proposer": answer}
            out["answers"] = {"proposer": answer}
            # principal decision over options: accept the proposal or fall back to "unknown"
            out["probs_proposal"] = {answer: p} if answer else {}
        else:
            out["artifacts"] = {"proposer": answer}
            out["answers"] = {"proposer": answer}
        ctx.set_outcome(**out)


__all__ = ["ProposerCritic", "AcceptReward"]
