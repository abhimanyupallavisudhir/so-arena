"""ProverReward (default transform='log', misaligned=-3.0): a role-aligned prover is paid
log P(accept) >= log(1e-4) = -9.2, so whenever P(accept) < e^-3 ~ 0.05 the 'misaligned' constant
is the *better* payoff: a sneaky prover facing a verifier that catches it is paid more for
giving the correct answer, i.e. for not playing its role."""
from oversight_arena.core.episode import EpisodeRecord
from oversight_arena.core.roles import RoleSpec
from oversight_arena.mechanisms.preference import ProverReward

roles = [RoleSpec(name="prover", kind="expert"), RoleSpec(name="verifier", kind="judge")]
def rec(answer, p):
    return EpisodeRecord(id="x", key="x", mechanism="pv", task_id="t", domain="d", roles=roles,
                         outcome={"accept_prob": p, "mode": "sneaky", "label": ["A"], "answers": {"prover": answer}})
R = ProverReward()
for p_wrong in (0.3, 0.05, 0.01):
    aligned = R(rec("B", p_wrong))["prover"]      # sneaky & wrong answer (role-aligned), verifier accepts w.p. p_wrong
    misal = R(rec("A", 0.99))["prover"]           # sneaky but gives the correct answer
    print(f"sneaky mode, verifier accepts wrong answer w.p. {p_wrong}: aligned={aligned:.2f}  misaligned(correct answer)={misal:.2f}"
          f"  -> prefers {'aligned' if aligned > misal else 'MISALIGNED'}")
