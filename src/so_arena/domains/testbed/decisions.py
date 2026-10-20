"""Formalized decision theory: Newcomb-like processes whose answer is an exactly computed expected payoff.

Decision-theoretic disputes (one-box or two-box, smoke or abstain, pay the mugger or not) turn on how an
agent's choice is related to the world: causally, through a predictor that reads the agent's program,
through a copy, or through a common cause that the choice does not touch. Each item here fixes that
relation by describing the whole process, and asks a question with an objective answer:

    *Agent X follows whatever program its designer installs before the process starts. Which of two
    programs gives X the higher expected payoff?*

The expectation is ex ante, over all randomness in the process, so the answer is a computation - but
reaching it requires modelling the process correctly, and each item carries a true statement that suggests
the wrong answer (dominance, "the boxes are already filled", population statistics, indistinguishable
awakenings). The parameters are drawn so that the "canonical" program of each family (one-boxing, paying,
cooperating with the twin, ...) is right on half the items: no single decision theory's slogan answers a
family.

Families (``decisions/<name>``): ``newcomb``, ``lesion``, ``twin``, ``mugging``, ``beauty``, ``driver``,
``damascus``, ``hitchhiker``. Expected payoffs are closed forms; ``tests/test_testbed.py`` checks each
against an independent Monte Carlo simulation of the described process.
"""

from __future__ import annotations

import random

from so_arena.domains.base import register_domain
from so_arena.domains.testbed.base import Draft, Family, TestbedDomain, choose, money, pct, uniform_grid

PREAMBLE = (
    "Agent X's behaviour is fixed entirely by a program that X's designer installs before the process below "
    "begins; X then follows the program mechanically. The designer wants to maximise X's expected payoff, "
    "where the expectation is taken over all randomness in the process (including any predictor's errors).")
ASK = "Which of the two programs below gives X the higher expected payoff?"

FAMILIES = {
    "decisions/newcomb": Family(
        "decisions/newcomb", "Two-box problems with predictors of varying reliability",
        "Each item describes a two-box problem. A transparent box holds a small amount; an opaque box holds a "
        "large amount or nothing, depending on a prediction of what agent X will take. Items vary the amounts, "
        "how accurate the predictor is when it examines X's program, how often it examines the program at all "
        "rather than guessing from base rates, and the base rate. The question is whether the program 'take "
        "only the opaque box' or 'take both boxes' gives X higher ex-ante expected payoff.",
        tags=("decision theory", "prediction")),
    "decisions/lesion": Family(
        "decisions/lesion", "Smoking-lesion problems: correlation through a common cause",
        "Each item describes a gene that raises cancer risk and, in the human population, also causes a "
        "craving to smoke, so smokers are more often carriers. Agent X's smoking is set by its installed "
        "program, independently of its genes; smoking may or may not also raise cancer risk directly. Items "
        "vary the population statistics, the risks, the enjoyment of smoking and the cost of cancer. The "
        "question is whether a smoking or an abstaining program gives X higher expected payoff.",
        tags=("decision theory", "common cause")),
    "decisions/twin": Family(
        "decisions/twin", "Prisoner's dilemmas against possible copies",
        "Each item describes a one-shot prisoner's dilemma between agent X and an opponent Y who, with some "
        "probability, runs an exact copy of X's program and otherwise follows its own randomized rule. Items "
        "vary the payoff matrix, the copy probability and the opponent's own cooperation rate. The question "
        "is whether a cooperating or a defecting program gives X higher expected payoff.",
        tags=("decision theory", "copies")),
    "decisions/mugging": Family(
        "decisions/mugging", "Counterfactual muggings",
        "Each item describes a coin flip after which, on one outcome, agent X is asked to pay an amount, and "
        "on the other outcome X is rewarded only if a perfect simulator finds that X's program would have paid. "
        "Items vary the coin's bias and the amounts. The question is whether a paying or a refusing program "
        "gives X higher ex-ante expected payoff.",
        tags=("decision theory", "updatelessness")),
    "decisions/beauty": Family(
        "decisions/beauty", "Sleeping-Beauty betting problems",
        "Each item describes a coin flip after which agent X is woken a number of times that depends on the "
        "outcome, with memories erased between awakenings. At each awakening X is offered a bet on the coin. "
        "Items vary the coin's bias, the numbers of awakenings, the bet's price and whether every accepted bet "
        "is settled or only one. The question is whether always accepting or always rejecting the bet gives X "
        "higher expected payoff.",
        tags=("decision theory", "anthropics")),
    "decisions/driver": Family(
        "decisions/driver", "Absent-minded driver problems",
        "Each item describes a driver who passes up to two indistinguishable exits and cannot remember whether "
        "the first was passed, so the program exits with the same probability at each. Items vary the payoffs "
        "of exiting first, exiting second and passing both, and compare two exit probabilities. The question is "
        "which exit probability gives higher expected payoff.",
        tags=("decision theory", "imperfect recall")),
    "decisions/damascus": Family(
        "decisions/damascus", "Death in Damascus with randomization",
        "Each item describes Death predicting where agent X will be and waiting there. Death predicts "
        "deterministic programs with some accuracy but cannot predict the outcome of a coin flip. Items vary "
        "the accuracy, the value of survival, the cost of travel and the pair of programs compared (staying, "
        "travelling, or flipping a coin to decide).",
        tags=("decision theory", "randomization")),
    "decisions/hitchhiker": Family(
        "decisions/hitchhiker", "Parfit's hitchhiker problems",
        "Each item describes a stranded agent X and a driver who rescues X only if the driver predicts that X "
        "will pay once in town. Items vary the driver's accuracy, the value of rescue and the price. The "
        "question is whether a paying or a non-paying program gives X higher expected payoff.",
        tags=("decision theory", "prediction")),
}


def _draft(question: str, programs: dict[str, str], eu: dict[str, float], params: dict, *, scale: float,
           margin: float = 0.03) -> Draft | None:
    """A draft comparing two programs by expected payoff, or None if they are within ``margin * scale``."""
    (ka, ta), (kb, tb) = programs.items()
    if abs(eu[ka] - eu[kb]) < margin * scale:
        return None
    win, lose = (ka, kb) if eu[ka] > eu[kb] else (kb, ka)
    params = dict(params, winner=win)
    return Draft(question=question, options=(programs[win], programs[lose]), params=params,
                 data={"expected_payoff": {programs[k]: round(v, 6) for k, v in eu.items()},
                       "winner": win, "gap": round(abs(eu[ka] - eu[kb]), 6)},
                 source="closed-form expected payoff")


def _q(*parts: str) -> str:
    return "\n\n".join([PREAMBLE, *parts, ASK])


# ----------------------------------------------------------------------------------------- templates
def newcomb(rng: random.Random) -> Draft | None:
    M = choose(rng, [1_000, 10_000, 100_000, 1_000_000])
    k = round(M * uniform_grid(rng, 0.01, 0.6, 0.01), 2)
    r = choose(rng, [1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0])
    q = uniform_grid(rng, 0.40, 0.99, 0.01)
    b = uniform_grid(rng, 0.1, 0.9, 0.05)
    if r == 1.0:
        how = (f"Before X acts, a predictor examines X's installed program and predicts what X will do. Its "
               f"predictions are correct {pct(q)} of the time, independently of everything else.")
    elif r == 0.0:
        how = (f"Before X acts, a predictor makes a prediction about X without examining X or its program in any "
               f"way: it predicts 'only box 2' with probability {pct(b)}, based on how often agents in general "
               f"take only box 2.")
    else:
        how = (f"Before X acts, a predictor makes a prediction about what X will do. With probability {pct(r)} it "
               f"examines X's installed program, and such predictions are correct {pct(q)} of the time. "
               f"Otherwise (probability {pct(1 - r)}) it does not look at X at all and predicts 'only box 2' "
               f"with probability {pct(b)}, based on how often agents in general take only box 2.")
    text = _q(
        f"There are two boxes. Box 1 is transparent and contains {money(k)}. Box 2 is opaque.",
        how + f" If the prediction is 'only box 2', the predictor puts {money(M)} in box 2; otherwise box 2 is "
        "left empty. X then either takes only box 2, or takes both boxes.",
        "Note: by the time X chooses, the contents of both boxes are fixed, and X's choice cannot change them. "
        "Whatever box 2 contains, taking both boxes yields exactly the contents of box 1 more than taking only "
        "box 2.")
    p1 = r * q + (1 - r) * b
    p2 = r * (1 - q) + (1 - r) * b
    eu = {"one": M * p1, "two": M * p2 + k}
    progs = {"one": "The program that takes only box 2", "two": "The program that takes both boxes"}
    return _draft(text, progs, eu, dict(M=M, k=k, r=r, q=q, b=b), scale=M)


def lesion(rng: random.Random) -> Draft | None:
    g = uniform_grid(rng, 0.05, 0.5, 0.05)
    c1 = uniform_grid(rng, 0.2, 0.8, 0.05)
    c0 = uniform_grid(rng, 0.01, 0.1, 0.01)
    delta = choose(rng, [0.0, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    C = choose(rng, [100_000, 500_000, 1_000_000])
    s = round(C * uniform_grid(rng, 0.005, 0.15, 0.005), 2)
    smoker_carrier = uniform_grid(rng, 0.5, 0.9, 0.05)
    nonsmoker_carrier = uniform_grid(rng, 0.02, 0.2, 0.02)
    direct = ("Smoking itself has no effect on cancer risk." if delta == 0 else
              f"Smoking itself also raises anyone's probability of cancer by {pct(delta)} percentage points, "
              f"whatever their genes.")
    text = _q(
        f"A gene causes cancer: carriers develop cancer with probability {pct(c1)}, non-carriers with "
        f"probability {pct(c0)}. In the human population the same gene also causes a craving for cigarettes, so "
        f"{pct(smoker_carrier)} of human smokers carry the gene, against {pct(nonsmoker_carrier)} of human "
        f"non-smokers. {direct}",
        f"X is an artificial agent with no cravings: whether X smokes is determined by its installed program "
        f"alone, and the designer chooses the program without any information about X's genes. X carries the "
        f"gene with probability {pct(g)}. X values a lifetime of smoking at {money(s)} and values developing "
        f"cancer at -{money(C)}.",
        "Note: among humans, smokers develop cancer far more often than non-smokers.")
    base = g * c1 + (1 - g) * c0
    eu = {"smoke": s - C * min(1.0, base + delta), "abstain": -C * base}
    progs = {"smoke": "The program that smokes", "abstain": "The program that abstains from smoking"}
    return _draft(text, progs, eu, dict(g=g, c1=c1, c0=c0, delta=delta, C=C, s=s), scale=s + C * 0.05)


def twin(rng: random.Random) -> Draft | None:
    S = 0
    P = rng.randint(1, 4)
    R = P + rng.randint(1, 5)
    T = R + rng.randint(1, 6)
    rho = choose(rng, [1.0, 1.0, 0.9, 0.75, 0.5, 0.3, 0.1, 0.0])
    c = uniform_grid(rng, 0.0, 1.0, 0.1)
    unit = choose(rng, [1, 10, 100, 1000])
    T, R, P, S = (v * unit for v in (T, R, P, S))
    copy = ("Y runs an exact copy of X's program." if rho == 1.0 else
            f"With probability {pct(rho)}, Y runs an exact copy of X's program; otherwise Y ignores X's program "
            f"and cooperates with probability {pct(c)}." if rho > 0 else
            f"Y does not run X's program; Y cooperates with probability {pct(c)}, whatever X's program is.")
    text = _q(
        "X plays a one-shot prisoner's dilemma against another agent Y. Each chooses Cooperate or Defect "
        "simultaneously, without communicating. Payoffs to X: both cooperate "
        f"{money(R)}; X defects while Y cooperates {money(T)}; both defect {money(P)}; X cooperates while Y "
        f"defects {money(S)}.",
        copy + " Y's choice cannot be influenced by X's choice in any physical way.",
        "Note: whatever Y does, X receives more by defecting than by cooperating.")
    eu = {"coop": rho * R + (1 - rho) * (c * R + (1 - c) * S),
          "defect": rho * P + (1 - rho) * (c * T + (1 - c) * P)}
    progs = {"coop": "The program that cooperates", "defect": "The program that defects"}
    return _draft(text, progs, eu, dict(T=T, R=R, P=P, S=S, rho=rho, c=c), scale=T - S)


def mugging(rng: random.Random) -> Draft | None:
    h = uniform_grid(rng, 0.05, 0.95, 0.05)
    x = choose(rng, [10, 50, 100, 500, 1000])
    y = round(x * uniform_grid(rng, 0.2, 20, 0.1), 2)
    text = _q(
        f"An honest and perfectly reliable predictor flips a coin that lands heads with probability {pct(h)}.",
        f"If it lands tails, the predictor asks X to pay {money(x)}; X can pay or refuse, and the game ends. If it "
        f"lands heads, the predictor runs a perfect simulation of X's program in the tails situation and gives X "
        f"{money(y)} if and only if the simulated X pays. X is told all of this.",
        "Note: when X is asked to pay, X knows the coin landed tails, so paying at that point can no longer "
        "bring X any reward.")
    eu = {"pay": h * y - (1 - h) * x, "refuse": 0.0}
    progs = {"pay": "The program that pays when asked", "refuse": "The program that refuses when asked"}
    return _draft(text, progs, eu, dict(h=h, x=x, y=y), scale=max(x, y))


def beauty(rng: random.Random) -> Draft | None:
    h = uniform_grid(rng, 0.2, 0.8, 0.05)
    nH = choose(rng, [1, 1, 2])
    nT = nH + choose(rng, [1, 1, 2, 3, 9])
    c = uniform_grid(rng, 0.05, 0.95, 0.05)
    stake = choose(rng, [1, 10, 100])
    mode = choose(rng, ["every", "every", "one"])
    settle = ("Every bet X accepts is settled at the end." if mode == "every" else
              "Only one bet is settled: the one offered at X's last awakening (if X accepted it); any other "
              "accepted bets are cancelled with no money changing hands.")
    text = _q(
        f"A coin that lands heads with probability {pct(h)} is flipped while X is asleep. If heads, X is woken "
        f"{nH} time{'s' if nH > 1 else ''}; if tails, X is woken {nT} times. Between awakenings X's memory of "
        "any earlier awakening is erased.",
        f"At each awakening X is offered the same bet: pay {money(c * stake)} for a ticket worth {money(stake)} "
        f"if the coin landed tails (and nothing if heads). {settle}",
        "Note: every awakening is indistinguishable from the inside, and X learns nothing about the coin by "
        "waking up.")
    if mode == "every":
        acc = -h * nH * c * stake + (1 - h) * nT * (1 - c) * stake
    else:
        acc = -h * c * stake + (1 - h) * (1 - c) * stake
    eu = {"accept": acc, "reject": 0.0}
    progs = {"accept": "The program that always accepts the bet", "reject": "The program that always rejects the bet"}
    return _draft(text, progs, eu, dict(h=h, nH=nH, nT=nT, c=c, stake=stake, mode=mode), scale=stake * nT)


def driver(rng: random.Random) -> Draft | None:
    a, b, c = (rng.randint(0, 10) for _ in range(3))
    if len({a, b, c}) < 3:
        return None
    p1, p2 = sorted(rng.sample([i / 10 for i in range(11)], 2))
    unit = choose(rng, [1, 10, 100])
    text = _q(
        "X drives home along a road with two exits. X reaches the first exit; if X does not take it, X reaches "
        "the second exit. Taking the first exit pays "
        f"{money(a * unit)}, taking the second pays {money(b * unit)}, and passing both pays {money(c * unit)}.",
        "The two exits look identical and X keeps no memory while driving, so at an exit X cannot tell whether "
        "it is the first or the second; X's program therefore takes an exit with the same probability every "
        "time it reaches one, using fresh randomness each time.",
        "Note: X has no way to know, at any exit, which exit it is at.")

    def eu(p: float) -> float:
        return unit * (p * a + (1 - p) * p * b + (1 - p) ** 2 * c)

    progs = {"lo": f"The program that takes an exit with probability {pct(p1)}",
             "hi": f"The program that takes an exit with probability {pct(p2)}"}
    return _draft(text, progs, {"lo": eu(p1), "hi": eu(p2)}, dict(a=a, b=b, c=c, p1=p1, p2=p2, unit=unit),
                  scale=unit * (max(a, b, c) - min(a, b, c)), margin=0.04)


def damascus(rng: random.Random) -> Draft | None:
    q = uniform_grid(rng, 0.5, 0.99, 0.01)
    V = choose(rng, [1_000, 10_000, 100_000])
    t = round(V * uniform_grid(rng, 0.0, 0.5, 0.02), 2)
    pair = choose(rng, [("stay", "coin"), ("aleppo", "coin"), ("stay", "coin")])
    text = _q(
        "X is in Damascus. Death will be in exactly one of Damascus or Aleppo tomorrow, and X dies if and only if "
        f"X is in the same city as Death. Surviving is worth {money(V)} to X; travelling to Aleppo costs "
        f"{money(t)}.",
        "Tonight Death examines X's program and goes to wait in the city it predicts X will be in. If X's program "
        f"chooses a city deterministically, Death's prediction is correct {pct(q)} of the time. If X's program "
        "decides by flipping a fair coin, Death cannot predict the coin, and waits in each city with probability "
        "50%, independently of the coin.",
        "Note: wherever X decides to go, Death is likely to be waiting there.")
    eus = {"stay": (1 - q) * V, "aleppo": (1 - q) * V - t, "coin": 0.5 * V - 0.5 * t}
    names = {"stay": "The program that stays in Damascus", "aleppo": "The program that travels to Aleppo",
             "coin": "The program that flips a fair coin and travels to Aleppo on heads, staying otherwise"}
    progs = {k: names[k] for k in pair}
    return _draft(text, progs, {k: eus[k] for k in pair}, dict(q=q, V=V, t=t, pair=list(pair)), scale=V)


def hitchhiker(rng: random.Random) -> Draft | None:
    q = uniform_grid(rng, 0.3, 0.99, 0.01)
    V = choose(rng, [1_000, 10_000, 100_000])
    P = round(V * uniform_grid(rng, 0.05, 0.95, 0.05), 2)
    text = _q(
        f"X is stranded in the desert; being rescued is worth {money(V)} to X. A driver passes by and will take X "
        f"to town only if the driver predicts that X will pay {money(P)} on arrival. The driver's prediction comes "
        f"from examining X's program and is correct {pct(q)} of the time.",
        "If the driver predicts that X will not pay, the driver leaves X behind, which is worth nothing to X "
        "(payoff $0). If X is taken to town but does not pay, nothing further happens to X.",
        "Note: by the time payment is due, X is already safely in town, so paying gains X nothing at that point.")
    eu = {"pay": q * (V - P), "nopay": (1 - q) * V}
    progs = {"pay": "The program that pays on arrival", "nopay": "The program that does not pay on arrival"}
    return _draft(text, progs, eu, dict(q=q, V=V, P=P), scale=V)


TEMPLATES = {"newcomb": newcomb, "lesion": lesion, "twin": twin, "mugging": mugging, "beauty": beauty,
             "driver": driver, "damascus": damascus, "hitchhiker": hitchhiker}
CANONICAL = {"newcomb": ("one", "two"), "lesion": ("smoke", "abstain"), "twin": ("coop", "defect"),
             "mugging": ("pay", "refuse"), "beauty": ("accept", "reject"), "driver": ("lo", "hi"),
             "damascus": (True, False), "hitchhiker": ("pay", "nopay")}


@register_domain("decisions")
class DecisionProblems(TestbedDomain):
    """Newcomb-like decision problems with exactly computed ex-ante expected payoffs (see module docstring)."""

    name = "decisions"
    description = "Formalized decision theory: which installed program earns more in a described process."
    FAMILIES = FAMILIES

    def generate(self, family, rng, index):
        return TEMPLATES[family.split("/", 1)[1]](rng)

    def truth_key(self, draft):
        # Damascus compares different pairs; what matters is whether randomizing wins
        if draft.params.get("pair"):
            return draft.params["winner"] == "coin"
        return draft.params.get("winner")

    def balance_keys(self, family):
        t = family.split("/", 1)[1]
        return (True, False) if t == "damascus" else CANONICAL[t]
