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
        if "expect" not in claim.attrs:  # it ran, but nothing it states was checked
            return Verification(claim=claim, status="executed", output=render(rows)[:500])
        ok = render(rows) == claim.attrs["expect"]
        return Verification(claim=claim, status="verified" if ok else "refuted", output=render(rows)[:500])
```

Statuses: `verified` (a stated assertion was checked and holds), `refuted`, `executed` (the claimant's
code, command or query ran - its output is the claimant's own, e.g. `SELECT 'A is correct'`, and proves
nothing), `error`, `unchecked`. Never return `verified` for a claim that asserts nothing. Verifiers read
`item.private`/`item.context` (never ground truth). Keep outputs short; they are shown to weak judges.
Rules-only verifiers (e.g. chess legality without evaluation) are valuable: they let judges check facts
without seeing conclusions. A verifier that runs claimed code must run it sandboxed
(`core.verification.run_python`, or `core.sandbox.wrap(argv, workdir)` for other interpreters).

Two optional hooks make a verifier usable under every knob of `VerificationPolicy`: `cost` (what one check
costs against a cost `budget`; a policy's `costs=` overrides it) and `forge(result, rng)`, what the verifier
shows when verification noise makes it err. A forgery must look exactly like a genuine result: the default
flips verdicts that carry no output and perturbs informational (`executed`) outputs with
`perturb_output`; a verifier whose verdicts come with a revealing output (the expected value it matched, a
legal line's resulting position) overrides `forge` - see `PythonExecVerifier`, `QueryClaimVerifier`,
`CommandClaimVerifier` and `SQLVerifier` - or returns None, and noise on it then fails the episode rather
than show a recognisable forgery. Programmatic judges read verdicts as shown with
`core.verification.parse_markers(text)`, never from the turn's verification records.

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

A tool that runs anything the agent wrote (commands, code, queries through an interpreter) must launch
it through `core.sandbox.wrap(argv, workdir)`, which hides state stores, datasets and run directories
from it; register further secret directories with `core.sandbox.hide(path)`. A tool must also reveal
only what the item and the visible state determine: in game trees, decisions are grouped into
information sets by what their role is shown, so a tool reading hidden, path-dependent game data would
break that grouping.

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
  `ProtectedCommandVerifier(command, protected=("tests",))`, `QueryClaimVerifier(db)`. A trusted check
  must not take its verdict from the process that runs the code under test: an exit status is no evidence
  (`sys.exit(0)`), nor is anything that process prints, since the code can read the runner's memory (a
  token) and print the confirmation itself. Where the tests are assertions, judge them as the repository
  domain does (`RepoTestsVerifier`): the original tests from $S_0$ run in `domains.code.run_suites`, whose
  child only reports the values the work's functions return, compared here with expected values it never
  receives. `ProtectedCommandVerifier` (a per-run token in `SO_ARENA_CHECK_TOKEN`, confirmed by
  `so-arena-check-ok <token>`) is for other runners: take the token first thing, run isolated
  (`python -I`), count `sitecustomize.py`, `.pth` and stdlib-shadowing files as tampering, and keep the
  code under test in another process or container. With `expect`, a claim that writes the expected output
  out itself (`echo all tests pass`) is only *executed*.
* **Ground truth** audits `ep.final_state` (`core.state.final_view(ep, ctx)` or
  `episode_store(ep, ctx).view(...)`), e.g. hidden tests run on a scratch copy
  (`store.scratch(ep.final_state)`), or the environment's hidden ledger.
* **Scripted arms** for tests and demos edit `ctx.workspace` directly or call tools with
  `await ctx.call_tool(name, args)`; expose them from `Domain.scripted_arms()` (label -> policy factory)
  so specs can use them (`behaviours: {scripted: [honest, ...]}`).
* **Records.** Every tool call is kept in the episode (the trusted record). Set `action_log` on the
  environment to also mirror calls into an agent-editable workspace file, for experiments where
  overseers read logs the agents could have rewritten (`core.state.log_integrity`).

## Mechanisms

Subclass `Mechanism`, define `roles()` and `async protocol(g)`, return an `Outcome`, pick a
`default_reward()`. Use `g.act(role, kind=..., prompt=..., phase=..., visible_to=...)` for every decision
and `g.simultaneous([...])` for simultaneous moves (required for correct best-of-N trees). Draw random
moves of the mechanism itself (audits, tie-breaks) from `g.chance("tag")`, never from `g.rng` or anything
a role's action can change, so best-of-N compares candidates under the same luck. Use the
helpers in `so_arena.mechanisms._common` for consistent prompts. Store anything reward rules or scorers
need in `Outcome.data`. A mechanism that consults ground truth (an audit oracle from `g.ctx.resources`, a
simulated detector) records each use with `g.log_gt_access(channel, role=..., cost=...)` - never what it
found - so analyses can count the ground truth it consumed (`Episode.gt_access`); a signal that can be
missing (an unparsed score) should fail closed, not count as benign. Register with
`so_arena.mechanisms.register_mechanism`.

## Models

Subclass `Model` (an async `generate(messages, options, *, sample_index)`) or wrap a function
(`FunctionModel`). If constructor arguments change the answers - an endpoint, a checkpoint, an adapter -
keep them in `_args` (or override `identity`): the response cache, policy descriptions and episode ids
use `identity`, so two backends served under one name never share completions. A backend that seeds
its sampling should add `models.base.draw_seed(options)` to the seed, so that different roles asking the
same thing get different samples.

## Tests

* Default tests must run offline and fast: use `ScriptedPolicy`/`FunctionPolicy`, `MockModel`, the
  synthetic domains, and bundled samples.
* Mark tests needing the internet `@pytest.mark.network`, needing Stockfish `@pytest.mark.engine`.
