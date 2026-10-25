import oversight_arena as oa
from oversight_arena.analysis import asd
from oversight_arena.analysis.games import EmpiricalGame
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
AG = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(claims=3)}
dom = HiddenBits(n_tasks=20)
verified, cheap = Debate(rounds=1, evidence=EvidencePolicy(budget=3)), Debate(rounds=1)
print("display names:", verified.display_name, cheap.display_name, "| hashes differ:", verified.config_hash() != cheap.config_hash())
res = oa.Experiment(dom, [verified, cheap], AG, oa.Stances(roles=["debater_a", "debater_b"]), progress=False).run()
print(asd(res, roles=["debater_a", "debater_b"])[["mechanism", "asd", "n_tasks", "n_episodes"]])
print(asd(res, roles=["debater_a", "debater_b"], by=("mechanism_hash",))[["mechanism_hash", "asd", "n_tasks", "n_episodes"]])
