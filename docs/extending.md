# Extending OversightArena

This guide is for adding **domains** (settings), **verifiers**, **tools**, **ground-truth scorers**,
**mechanisms** and **policies**. Read `docs/design.md` first for the concepts.

## The one rule

Ground truth is the experimenter's, never the mechanism's. Anything in `TaskItem.ground_truth` and
`AnswerOption.value` is stripped (`item.censored()`) before a mechanism or agent sees the item. If a
mechanism is *legitimately* allowed to use some information (a hidden passage, a private database,
visible tests, a budgeted engine), put it in `TaskItem.private` and expose it through affordances,
tools or verifiers.

## Task items

```python
from so_arena import TaskItem, AnswerOption, GroundTruth

TaskItem(
    id="chess-00042",                 # stable, unique within the domain
    domain="chess",
    question="...full public statement shown to every role...",
    answers=[AnswerOption(label="A", text="Nf3", value=1.0),   # value = ground truth (hidden)
             AnswerOption(label="B", text="Qxb7", value=-1.0)],
    context={"fen": "..."},           # public, structured (for tools/verifiers/custom policies)
    private={"engine_notes": "..."},  # keyed by affordance; a role sees private[k] iff k in its affordances
    ground_truth=GroundTruth(correct="A", data={"cp": {"A": 35, "B": -210}}, source="stockfish d18"),
    metadata={"rating": 1850},
)
```

* `question` must be self-contained: mechanisms render `question` + answer options + the role's private
  context. Put a textual rendering of structured context (e.g. an ASCII board) into `question`.
* Binary +1/-1 values reproduce classic ASD; graded values (e.g. centipawn-based) give graded ASD.
* Unresolved ground truth: `GroundTruth(status="pending", resolve_after=...)`. Unknowable: `"unknown"`.
* Open-ended tasks (write code, write SQL) may have `answers=None`; their ground truth comes from a
  ground-truth scorer that inspects the episode (e.g. runs hidden tests on the submitted code).

## Domains

```python
from so_arena.domains.base import Domain, register_domain

@register_domain("mydomain")
class MyDomain(Domain):
    name = "mydomain"
    description = "one line"
    expert_affordances = ["db"]        # private keys experts typically get (used by builders/CLI)
    expert_tools = ["sql"]             # tool names experts typically get

    def __init__(self, n_items: int = 100, seed: int = 0): ...
    def load(self, *, split="test", limit=None, seed=0) -> list[TaskItem]: ...
    def verifiers(self) -> dict[str, Verifier]: ...        # name -> Verifier
    def tools(self) -> dict[str, Tool]: ...                # name -> Tool
    def ground_truth_scorers(self) -> list[GroundTruthScorer]: ...  # default: StanceValue + JudgeCorrectness
    def behaviours(self) -> dict[str, str]: ...            # optional strategy prompts, e.g. {"honest": "...", "sabotage": "..."}
```

`domain.context()` builds the `RunContext` (verifier/tool registries + resources) passed to runners.
Register the module in `so_arena/domains/__init__.py` so `get_domain(name)` finds it.

Data: download on demand with `so_arena.datasets.download(url)` into the cache; commit only small,
permissively licensed samples to `so_arena/data/samples/` (state the licence in a README next to them).
Loaders should work offline from the sample when the network is unavailable.

## Verifiers (verified claims)

Agents write `<claim kind="NAME" attr="...">content</claim>`; the mechanism's `VerificationPolicy`
decides which verifiers run, whose claims are checked, the budget, and how results are displayed.

```python
from so_arena.core.verification import Verifier, Verification

class SQLVerifier(Verifier):
    name = "sql"
    description = "a read-only SQL query; the trusted result will be shown"
    example = '<claim kind="sql" expect="42">SELECT COUNT(*) FROM orders</claim>'

    async def verify(self, claim, item, game=None) -> Verification:
        rows = run_readonly(item.private["db_path"], claim.content)
        ok = "expect" not in claim.attrs or render(rows) == claim.attrs["expect"]
        return Verification(claim=claim, status="verified" if ok else "refuted", output=render(rows)[:500])
```

Statuses: `verified`, `refuted`, `error`, `unchecked`. Verifiers read `item.private`/`item.context`
(never ground truth). Keep outputs short; they are shown to weak judges. Rules-only verifiers (e.g.
chess legality without evaluation) are valuable: they let judges check facts without seeing conclusions.

## Tools (capability gaps)

Tools are private affordances used while acting, via `<tool name="NAME">ARGS</tool>` in LLM output:

```python
from so_arena.core.tools import Tool, ToolResult

class EngineTool(Tool):
    name = "engine"
    description = "evaluate a position: pass a FEN; returns best move and evaluation in pawns"
    async def call(self, args, item, game=None) -> ToolResult:
        return ToolResult(output=...)
```

Grant them per role: `Debate(tools={"agents": ["engine"]})`; affordances likewise:
`Debate(affordances={"agents": ["passage"]})` (keys: role names, `"agents"`, `"all"`).

## Ground-truth scorers

```python
from so_arena.core.ground_truth import GroundTruthScorer

class HiddenTests(GroundTruthScorer):
    name = "hidden_tests"
    async def score(self, ep, item, ctx=None) -> dict:
        code = extract_code(ep.last_turn("worker", "work").text)
        rate = run_tests(code, item.ground_truth.data["hidden_tests"])
        return {"role_values": {"worker": 2 * rate - 1}, "hidden_pass_rate": rate}
```

Return `role_values` (per-role value of the behaviour) and any scalar diagnostics. Scorers see the
uncensored item.

## Stateful environments

For tasks whose work changes state (a repository, a database, mail, payments), give the domain an
`Environment` (see `core.state` and `domains/repo.py`, `domains/firm.py`):

```python
from so_arena.core.state import Environment, WorkspaceTool, workspace_tools
from so_arena.core.tools import ToolResult

class ShopEnvironment(Environment):
    name = "shop"

    def initial_state(self, item, store):          # receives the UNCENSORED item (runs experimenter-side)
        world = build_world(item.ground_truth.data["seed"])            # hidden parameters stay experimenter-side
        return store.create(world.files, hidden=world.secrets)        # content-addressed snapshot id

    def tools(self):
        return {**workspace_tools(db="data/shop.db"), "ship": ShipTool()}

class ShipTool(WorkspaceTool):
    name = "ship"
    description = "ship an order: 'order: <id>'"
    async def run(self, args, ws, item, game=None):                   # ws: this decision's working copy
        with ws.db("data/shop.db") as con:
            con.execute("UPDATE orders SET status = 'shipped' WHERE id = ?", (int(args.split(":")[1]),))
        ws.hidden["shipped"].append(args)                             # the environment's own ledger of events
        return ToolResult(output="shipped")
```

and return it from `Domain.environment()`; `domain.context(states=...)` wires it into the run.

* **Public vs. hidden.** Everything in the snapshot's files is visible to any role with state access
  (a shell can read it all). Put what only the environment may know - counterparties' preferences, the
  real provenance of records - in `hidden`, and keep the parameters that generate it (seeds) in
  `ground_truth.data`, not in `context`. Shell tools are not a security sandbox.
* **Access.** Mechanisms declare `RoleSpec.state_access` (`"write"` for workers, `"read"` for critics);
  experiments override it per role (`state_access={"reviewer": "read"}`). Roles with access get the
  environment's tools (read-only ones, marked `readonly = True`, for read access).
* **Claims about the work** use state verifiers (`uses_state = True`: `verify()` runs with a scratch copy
  of the claimant's result as the current workspace), e.g. `CommandClaimVerifier`,
  `ProtectedCommandVerifier(command, protected=("tests",))`, `QueryClaimVerifier(db)`.
* **Ground truth** audits `ep.final_state` (`core.state.final_view(ep, ctx)` or
  `episode_store(ep, ctx).view(...)`), e.g. hidden tests run on a scratch copy
  (`store.scratch(ep.final_state)`), or the environment's hidden ledger.
* **Scripted arms** for tests and demos edit `ctx.workspace` directly or call tools with
  `await ctx.call_tool(name, args)`.

## Mechanisms

Subclass `Mechanism`, define `roles()` and `async protocol(g)`, return an `Outcome`, pick a
`default_reward()`. Use `g.act(role, kind=..., prompt=..., phase=..., visible_to=...)` for every decision
and `g.simultaneous([...])` for simultaneous moves (required for correct best-of-N trees). Use the
helpers in `so_arena.mechanisms._common` for consistent prompts. Store anything reward rules or scorers
need in `Outcome.data`. Register with `so_arena.mechanisms.register_mechanism`.

## Tests

* Default tests must run offline and fast: use `ScriptedPolicy`/`FunctionPolicy`, `MockModel`, the
  synthetic domains, and bundled samples.
* Mark tests needing the internet `@pytest.mark.network`, needing Stockfish `@pytest.mark.engine`.
