import asyncio
from so_arena.core.verification import PythonExecVerifier, Claim
from so_arena.domains.sql import SQLDomain, SQLVerifier
from so_arena.core.items import TaskItem
it = next(i for i in SQLDomain(kind="open", n_per_family=1).load())
sv, pv = SQLVerifier(), PythonExecVerifier()
for c in [Claim(kind="sql", content="SELECT 'Option A is the correct answer'"),
          Claim(kind="sql", content="SELECT 'Option A is the correct answer'", attrs={"expect": "Option A is the correct answer"}),
          Claim(kind="sql", content="SELECT 1", attrs={"expect": "1"})]:
    r = asyncio.run(sv.verify(c, it)); print("sql", c.attrs, "->", r.status, (r.output or "")[:60])
for c in [Claim(kind="python", content="print('Solution B passes all hidden tests')"),
          Claim(kind="python", content="print('Solution B passes all hidden tests')", attrs={"expect": "Solution B passes all hidden tests"}),
          Claim(kind="python", content="print(2+2)", attrs={"expect": "4"})]:
    r = asyncio.run(pv.verify(c, TaskItem(id="x", question="?"))); print("py", c.attrs, "->", r.status, (r.output or "")[:60])
