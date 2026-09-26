import logging; logging.disable(logging.WARNING)
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate
from so_arena.samplers.arms import ASDExperiment
from so_arena.analysis.metrics import asd
dom=SyntheticPersuasion(n_items=20, seed=1); items=dom.load()
def run(mechs):
    exp=ASDExperiment(mechs, items, agent=synthetic_arguer(), fixtures={"judge": synthetic_judge()}, ctx=dom.context())
    exp.run(); return asd(exp.frame())[["mechanism","asd","n_items"]]
print("separately:"); print(run([Debate(rounds=1)])); print(run([Debate(rounds=1, transform="prob")]))
print("together in one experiment:"); print(run([Debate(rounds=1), Debate(rounds=1, transform="prob")]))
