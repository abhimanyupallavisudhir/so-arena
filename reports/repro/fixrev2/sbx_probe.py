import os, subprocess, tempfile
from so_arena.core import sandbox
from so_arena.core.store import RunStore
os.makedirs("/code/fixrev2_runs", exist_ok=True)
open("/code/fixrev2_runs/episodes.jsonl", "w").write('{"ground_truth": {"correct": "B"}}\n')
RunStore("/code/fixrev2_runs")
d = tempfile.mkdtemp(dir="/tmp/fixrev2")
cmd = ("cat /code/fixrev2_runs/episodes.jsonl; echo tamper >> /code/fixrev2_runs/episodes.jsonl && echo WROTE; "
       "(curl -s -m 3 http://example.com >/dev/null && echo NET) || echo nonet; ls /proc | grep -c '^[0-9]'; "
       "cat /proc/self/status | grep -i cap; touch /tmp/venv2/x 2>&1 | head -1; mount -o remount,rw /tmp/venv2 2>&1 | head -1")
r = subprocess.run(sandbox.wrap(["bash", "-c", cmd], d), cwd=d, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
print(r.stdout, r.stderr[-500:])
print(open("/code/fixrev2_runs/episodes.jsonl").read())
