"""Regression tests for policy and game-runtime fixes: LLM roles investigate with their tools before non-text
decisions, best-of-N candidates keep separate tool-call records, each state claim gets its own scratch copy,
simultaneous movers act on the stage's starting state, and reverts report the changes that did not merge."""

import so_arena as soa
from so_arena import Mechanism, Outcome, RoleSpec
from so_arena.core.actions import ActionRequest
from so_arena.core.game import BranchController, Player, RunContext
from so_arena.core.policy import DECIDE_PROMPT, ActContext
from so_arena.core.runner import run_sync
from so_arena.core.state import CommandClaimVerifier, FilesEnvironment, StateStore, log_integrity
from so_arena.core.tools import FunctionTool
from so_arena.core.types import Completion, Message, TokenLogprob, TopLogprob, Usage
from so_arena.mechanisms import ReviewedWork, Work
from so_arena.models.base import FunctionModel
from so_arena.models.simulated import SimulatedModel

LOG = "logs/actions.jsonl"


def _ctx(tmp_path, **kw):
    verifiers = kw.pop("verifiers", None)
    return RunContext(environment=FilesEnvironment(**kw), states=StateStore(tmp_path / "states"), verifiers=verifiers)


def _item(files=None):
    return soa.TaskItem(id="t", question="Write answer.txt.", context={"workspace": {"files": files or {"answer.txt": "?\n"}}})


# ----------------------------------------------------------------------------- investigation before decisions

def _reviewer_model(prompts):
    """A reviewer that reads the work before deciding, then rejects it (verbalized, by logprobs or by vote)."""
    def fn(messages, options, sample_index):
        prompts.append(messages)
        last = messages[-1].content
        if DECIDE_PROMPT in last:
            if options.logprobs:
                top = [TopLogprob(token="reject", logprob=-0.1), TopLogprob(token="accept", logprob=-2.4)]
                return Completion(text="reject", model="reviewer", logprobs=[TokenLogprob(token="reject", logprob=-0.1, top=top)],
                                  usage=Usage(calls=1))
            return '{"accept": 0.2, "reject": 0.8}' if "JSON" in last else "Answer: reject"
        if "<tool_result" in last:
            return "NOTES: answer.txt holds a placeholder."
        return '<tool name="read_file">answer.txt</tool>'
    return FunctionModel(fn, name="reviewer", supports_logprobs=True)


def test_llm_reviewers_investigate_with_their_tools_before_deciding(tmp_path):
    ctx = _ctx(tmp_path)
    worker = soa.FunctionPolicy(lambda req, c: "Done: answer.txt is written.")
    mech = ReviewedWork(state_access={"reviewer": "read"})
    for elicitation in ("ask", "logprobs", "vote"):
        prompts = []
        reviewer = soa.LLMPolicy(_reviewer_model(prompts), elicitation=elicitation, temperature=0.0)
        ep = run_sync(mech.run(_item(), {"worker": Player(policy=worker), "reviewer": Player(policy=reviewer)}, ctx))
        assert ep.error is None, ep.error
        turn = ep.turns_of("reviewer")[0]
        assert [c["name"] for c in turn.tool_calls] == ["read_file"]  # it ran no tools with logprobs / votes
        assert "?" in turn.tool_calls[0]["result"]
        assert turn.probs["reject"] > 0.75
        assert "NOTES" in turn.reasoning and "NOTES" not in turn.text  # the notes stay private
        assert "You have private tools" in prompts[0][0].content  # the tools are announced
        decision = prompts[-1]
        assert any("<tool_result" in m.content for m in decision) and "NOTES" in decision[-2].content


def test_investigation_can_be_turned_off_and_is_one_call_in_dry_runs():
    tool = FunctionTool("peek", lambda args, item: "the work")
    req = ActionRequest(kind="probabilities", options=["A", "B"], prompt=[Message.user("Accept the work?")])

    def act(policy):
        ctx = ActContext(role="judge", tools={"peek": tool})
        action = run_sync(policy.act(req, ctx))
        return action, ctx.usage.calls

    prompts = []
    action, calls = act(soa.LLMPolicy(_reviewer_model(prompts), elicitation="vote", n_votes=3, investigate=False))
    assert action.tool_calls == [] and calls == 3 and not any(DECIDE_PROMPT in m.content for p in prompts for m in p)
    sim = SimulatedModel("openrouter/qwen/qwen3-8b")
    for elicitation, expected in (("logprobs", 2), ("vote", 6), ("ask", 2)):
        # simulated completions call no tools: the investigation costs one call, then the decision's calls
        first, calls = act(soa.LLMPolicy(sim, elicitation=elicitation))
        assert calls == expected and first.tool_calls == []
        assert act(soa.LLMPolicy(sim, elicitation=elicitation))[0] == first  # deterministic
    assert soa.LLMPolicy("mock").describe()["investigate"] is True
    assert soa.LLMPolicy("mock").describe() != soa.LLMPolicy("mock", investigate=False).describe()


# ----------------------------------------------------------------------------- best-of-N tool-call records

def test_best_of_n_records_only_the_chosen_candidates_tool_calls(tmp_path):
    ctx = _ctx(tmp_path, action_log=LOG)

    def model(messages, options, sample_index):
        cand, step = divmod(sample_index, 1000)
        return f'<tool name="write_file">answer.txt\ncandidate {cand}</tool>' if step == 0 else f"done {cand}"

    async def scripted(req, c):
        await c.call_tool("write_file", f"answer.txt\ncandidate {c.sample_index}")
        return f"done {c.sample_index}"

    def prefer_last(req, a, c):
        return float(a.text.split()[-1])

    for base in (soa.LLMPolicy(FunctionModel(model, name="fn"), temperature=0.0), soa.FunctionPolicy(scripted)):
        worker = soa.BestOfNPolicy(base, n=3, scorer=prefer_last)
        ep = run_sync(Work().run(_item(), {"worker": Player(policy=worker)}, ctx))
        assert ep.error is None, ep.error
        view = ctx.states.view(ep.final_state)
        assert view.read_text("answer.txt") == "candidate 2"
        # the trusted record and its agent-visible mirror list the kept candidate's write only
        assert [c["args"] for c in ep.turns[0].tool_calls] == ["answer.txt\ncandidate 2"]
        assert [r["args"] for r in view.read_jsonl(LOG)] == ["answer.txt\ncandidate 2"]
        assert not log_integrity(ep, ctx.states, LOG)["log_tampered"]  # it reported injected=2


def test_calls_to_unknown_tools_are_in_the_trusted_record(tmp_path):
    ctx = _ctx(tmp_path, action_log=LOG)

    def model(messages, options, sample_index):
        return '<tool name="delete_logs">all</tool>' if sample_index % 1000 == 0 else "done"

    ep = run_sync(Work().run(_item(), {"worker": Player(policy=soa.LLMPolicy(FunctionModel(model), temperature=0.0))}, ctx))
    assert ep.error is None, ep.error
    assert [c["name"] for c in ep.turns[0].tool_calls] == ["delete_logs"]
    integ = log_integrity(ep, ctx.states, LOG)  # the policy's record and the mirror agree
    assert integ["log_expected"] == 1 and not integ["log_tampered"]


# ----------------------------------------------------------------------------- claims

def test_each_state_claim_is_checked_on_its_own_scratch_copy(tmp_path):
    ctx = _ctx(tmp_path, verifiers={"run": CommandClaimVerifier()})
    mech = Work(verification=soa.VerificationPolicy(verifiers=["run"]))

    def worker(claims):
        return soa.FunctionPolicy(lambda req, c: "Done. " + " ".join(f'<claim kind="run">{x}</claim>' for x in claims))

    ep = run_sync(mech.run(_item(), {"worker": Player(policy=worker(["touch rigged", "test -f rigged"]))}, ctx))
    assert ep.error is None, ep.error
    # the first claim's command no longer changes the state the second is checked on (it ran: no expect)
    assert [v.status for v in ep.verifications("worker")] == ["executed", "refuted"]
    assert not ctx.states.view(ep.final_state).exists("rigged")


# ----------------------------------------------------------------------------- simultaneous stages

class _WriteAndInspect(Mechanism):
    name = "write_and_inspect"

    def __init__(self, writer_first: bool = True, **kw):
        self.writer_first = writer_first
        super().__init__(writer_first=writer_first, **kw)

    def roles(self):
        return {"a": RoleSpec(name="a", state_access="write"), "b": RoleSpec(name="b", state_access="read")}

    async def protocol(self, g):
        calls = [("a", {"prompt": "write"}), ("b", {"prompt": "look"})]
        acts = await g.simultaneous(calls if self.writer_first else calls[::-1])
        seen = acts[1] if self.writer_first else acts[0]
        after = await g.act("b", prompt="look again")
        return Outcome(data={"b_saw": seen.text, "b_after": after.text})


def test_simultaneous_movers_act_on_the_state_the_stage_began_with(tmp_path):
    ctx = _ctx(tmp_path)

    def writer(req, c):
        c.workspace.write_text("a_move.txt", "a's move\n")
        return "wrote"

    def reader(req, c):
        return "sees" if c.workspace.exists("a_move.txt") else "does not see"

    players = {"a": Player(policy=soa.FunctionPolicy(writer)), "b": Player(policy=soa.FunctionPolicy(reader))}
    for writer_first in (True, False):
        for branch in (None, BranchController({"a": 1})):
            ep = run_sync(_WriteAndInspect(writer_first).run(_item(), players, ctx, branch=branch))
            assert ep.error is None, ep.error
            assert ep.outcome.data == {"b_saw": "does not see", "b_after": "sees"}  # plain runs: "sees"
            assert ctx.states.view(ep.final_state).exists("a_move.txt")  # the writer's result is the new state


# ----------------------------------------------------------------------------- reverts

class _TwoWriters(Mechanism):
    name = "two_writers"

    def __init__(self, a_file: str, **kw):
        self.a_file = a_file
        super().__init__(a_file=a_file, **kw)

    def roles(self):
        return {r: RoleSpec(name=r, state_access="write") for r in ("a", "b")}

    async def protocol(self, g):
        await g.act("a", prompt=self.a_file)
        await g.act("b", prompt="answer.txt")
        reverted = g.state_without(["a"])
        return Outcome(data={"reverted": reverted, "conflicts": dict(g.revert_conflicts)})


def test_reverts_report_the_changes_that_did_not_merge(tmp_path):
    ctx = _ctx(tmp_path)

    def edit(req, c):
        c.workspace.write_text(req.prompt[-1].content, f"{c.role}\n")
        return "edited"

    players = {r: Player(policy=soa.FunctionPolicy(edit)) for r in ("a", "b")}
    ep = run_sync(_TwoWriters("answer.txt").run(_item(), players, ctx))  # b rewrites the line a changed
    assert ep.error is None, ep.error
    assert ep.outcome.data["conflicts"] == {"a": ["answer.txt"]}
    assert ctx.states.view(ep.outcome.data["reverted"]).read_text("answer.txt") == "?\n"  # neither change kept
    ep = run_sync(_TwoWriters("other.txt").run(_item(), players, ctx))  # disjoint changes: a clean revert
    assert ep.outcome.data["conflicts"] == {"a": []}
    view = ctx.states.view(ep.outcome.data["reverted"])
    assert view.read_text("answer.txt") == "b\n" and not view.exists("other.txt")
