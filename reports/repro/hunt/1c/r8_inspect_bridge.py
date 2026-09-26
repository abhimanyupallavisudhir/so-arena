"""Inspect bridge: (a) batch reward rules (CA/DMI) are never computed -> no reward/<role> scores;
(b) results_from_logs concatenates every log in a directory without de-duplicating episode keys,
so re-running the eval (or eval-retry) into the same log dir double-counts episodes."""
import shutil, tempfile, logging
logging.disable(logging.WARNING)
from inspect_ai import eval as ieval
import oversight_arena as oa
from oversight_arena.integrations.inspect import oversight_task, results_from_logs
from oversight_arena.mechanisms.peer_prediction import Reporters, CorrelatedAgreement
from oversight_arena.domains.synthetic import HiddenBits
dom_cls = HiddenBits
dom = HiddenBits(n_tasks=6)
agent = oa.ScriptedAgent(lambda obs: {"choice": obs.task.option_ids[0]}, id="first")
logdir = tempfile.mkdtemp()
task = oversight_task(dom, Reporters(n=3, reward=CorrelatedAgreement()), agents={"*": agent})
for _ in range(2):  # e.g. the user re-runs the eval
    ieval(task, model="mockllm/model", log_dir=logdir, display="none")
log = results_from_logs(logdir)
print("domain:", dom_cls.__name__, "tasks:", len(dom.tasks()))
print("records loaded:", len(log), "unique keys:", len({r.key for r in log.records}))
print("rewards on first record:", log.records[0].rewards)
shutil.rmtree(logdir)
