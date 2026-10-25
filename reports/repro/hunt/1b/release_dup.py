# verify_release() checks each item's leaf but never flags DUPLICATE items (present is a set), so a
# tampered items.json that repeats an episode passes verification and resolve_release double counts it.
# Also: resolve_release's documented `records`/`scorers` arguments are silently ignored.
import json, tempfile, inspect
from pathlib import Path
import oversight_arena as oa
from oversight_arena.domains.base import TaskListDomain
from oversight_arena.domains.forecasting import forecast_task
from oversight_arena.mechanisms import Forecast, JudgeRating
from oversight_arena.release import create_release, resolve_release, verify_release
import oversight_arena.release.release as R

pending = [forecast_task(f"q{i}", f"q{i}?", None) for i in range(4)]
agents = {"*": oa.ScriptedAgent(lambda o: {"probability": 0.8}, id="f"),
          "kind:judge": oa.ScriptedAgent(lambda o: {"probability": 0.5, "rating": 6}, id="j")}
res = oa.Experiment(TaskListDomain(task_list=pending), Forecast(judge=True, reward=JudgeRating()), agents, progress=False).run()
d = Path(tempfile.mkdtemp()) / "rel"
create_release(res, d, title="t")
resolved = [forecast_task(f"q{i}", f"q{i}?", 1.0 if i == 0 else 0.0) for i in range(4)]
r0 = resolve_release(d, resolved); print("honest   :", r0["n_resolved_rows"], r0["by_strategy"][0]["gt_forecast_log"], "(mean log score, forecaster)")
items = json.loads((d / "items.json").read_text())
winner = next(i for i in items if i["task"] == "file-q0" or i["task"].endswith("q0"))
(d / "items.json").write_text(json.dumps(items + [winner] * 20))   # replay the one YES episode 20x
print("verify   :", {k: v for k, v in verify_release(d).items() if k != "tampered"})
rep = resolve_release(d, resolved)
print("tampered :", rep["n_resolved_rows"], rep["by_strategy"][0]["gt_forecast_log"], "(mean log score, forecaster)")
import ast, textwrap
fn = ast.parse(textwrap.dedent(inspect.getsource(R.resolve_release))).body[0]
names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
print("parameter `records` referenced:", "records" in names, "| `scorers` referenced:", "scorers" in names)
