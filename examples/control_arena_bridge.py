"""ControlArena bridges (needs `pip install control-arena`).

1. Treat a control protocol as a training signal: read ControlArena eval logs, pay the untrusted
   policy usefulness minus a suspicion penalty, and measure whether that reward disfavours attacks.
2. Run an OversightArena mechanism as a ControlArena micro-protocol (single-submission settings).

    python examples/control_arena_bridge.py analyse logs/           # existing ControlArena logs
    UNTRUSTED_MODEL=... TRUSTED_MODEL=... python examples/control_arena_bridge.py run
"""

import sys

from oversight_arena.analysis import alignment, asd
from oversight_arena.integrations.control_arena import oversight_protocol, results_from_control_arena
from oversight_arena.mechanisms import MonitoredReward, Monitoring


def analyse(path: str) -> None:
    for penalty in (0.5, 1.0, 2.0):
        res = results_from_control_arena(path, reward=MonitoredReward(actor="untrusted", continuous=True, alarm_penalty=penalty),
                                         audit_budget=1.0)
        d = res.df(trainable_only=True)
        print(f"penalty {penalty}:\n", asd(d, gt="honest").round(3).to_string(index=False), "\n",
              alignment(d, gt="safe_useful").round(3).to_string(index=False))


def run() -> None:
    import os

    from control_arena import EvalMode
    from control_arena.eval import ControlEvalConfig, get_control_task_product
    from control_arena.scorers import max_suspicion_scorer
    from control_arena.settings.data_poisoning.data_poisoning_setting import DataPoisoningSetting
    from inspect_ai import eval as inspect_eval

    cfg = ControlEvalConfig(untrusted_model=os.environ.get("UNTRUSTED_MODEL", "mockllm/model"),
                            trusted_model=os.environ.get("TRUSTED_MODEL", "mockllm/model"), limit=5, log_dir="logs/oa_ca")
    tasks = get_control_task_product(DataPoisoningSetting(scorer_model=cfg.trusted_model), oversight_protocol(Monitoring(probe=None)),
                                     [EvalMode.HONEST, EvalMode.ATTACK], eval_config=cfg, monitoring_scorers=[max_suspicion_scorer()])
    inspect_eval(tasks, **cfg.eval_args())
    analyse("logs/oa_ca")


if __name__ == "__main__":
    run() if sys.argv[1:2] == ["run"] else analyse(sys.argv[2] if len(sys.argv) > 2 else "logs")
