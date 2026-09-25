"""Human judges in the browser: a small rating server and a policy that routes decisions to it.

:class:`HumanPolicy` makes any role - a judge, a reviewer, a monitor - a person. Each decision becomes
a *rating task* on a :class:`RatingServer` (a local web app, standard library only): a rater opens
the page, reads what the role may see (its instructions included), answers with the form the request
needs (probabilities over the options, a choice, a score, text, or one field per key of a JSON answer)
and submits. The time between opening and submitting is recorded as oversight effort
(``Usage.effort_seconds``), and a per-policy **time budget** is shown as a countdown - and, with
``enforce_budget=True``, what the rater has entered is submitted when it runs out (anything left open
is an abstention) - so the evaluation budget is an experimental variable: run the same items with 2-,
10- and 30-minute judges and plot ASD against the budget (the Kinniment curve).

Every judgment is appended to ``judgments.jsonl`` in the server's directory (rater, budget, time
spent, answer). By default the server listens on localhost only; ``token=`` protects it when it must
be reachable by remote raters.

Example::

    server = RatingServer(port=8765, log_dir="runs/human")      # open http://localhost:8765
    judges = {b: HumanPolicy(server, time_budget_s=b, label=f"human-{b // 60}min") for b in (120, 600)}
    for b, judge in judges.items():
        ASDExperiment([Consultancy(), Debate()], items, agent=expert, fixtures={"judge": judge}).run()
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from so_arena.core.actions import Action, ActionRequest
from so_arena.core.policy import (
    ActContext,
    Policy,
    coerce_action,
    finalize_from_text,
    format_instructions,
    neutral_action,
)
from so_arena.core.types import Usage


@dataclass
class RatingTask:
    id: str
    kind: str
    title: str
    messages: list[dict[str, str]]
    options: list[str] = field(default_factory=list)
    option_texts: dict[str, str] = field(default_factory=dict)
    score_range: tuple[float, float] | None = None
    score_meaning: str | None = None
    budget_s: float | None = None
    enforce_budget: bool = False
    created_at: float = field(default_factory=time.time)
    opened_at: float | None = None
    rater: str | None = None
    result: dict[str, Any] | None = None
    loop: Any = None
    future: Any = None
    json_keys: list[str] = field(default_factory=list)  # kind "json": the page shows one field per key
    expand_instructions: bool = False  # show the role's instructions (system messages) unfolded
    cancelled: bool = False  # nobody awaits the answer any more (the run was interrupted)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "messages": self.messages,
                "options": self.options, "option_texts": self.option_texts,
                "score_range": list(self.score_range) if self.score_range else None,
                "score_meaning": self.score_meaning, "budget_s": self.budget_s, "enforce_budget": self.enforce_budget,
                "opened_at": self.opened_at, "json_keys": self.json_keys,
                "expand_instructions": self.expand_instructions}


class RatingConflict(Exception):
    """A judgment the server cannot record (HTTP 409): the task is answered, cancelled or another rater's."""


class RatingServer:
    """A queue of rating tasks served to raters over HTTP (``/``: the rating page)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, *, log_dir: str | Path | None = None,
                 token: str | None = None, reassign_after_s: float | None = 3600.0):
        self.token = token
        self.reassign_after_s = reassign_after_s
        self.log_path = Path(log_dir) / "judgments.jsonl" if log_dir is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, RatingTask] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._done = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _authorized(self, query: dict[str, list[str]]) -> bool:
                if server.token is None:
                    return True
                given = self.headers.get("X-Token") or (query.get("token") or [""])[0]
                return secrets.compare_digest(given, server.token)

            def _send(self, code: int, body: bytes = b"", ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if not self._authorized(q):
                    return self._send(HTTPStatus.FORBIDDEN, b'{"error": "bad token"}')
                if url.path in ("/", "/index.html"):
                    return self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
                if url.path == "/api/next":
                    task = server.next_task((q.get("rater") or ["anonymous"])[0][:80])
                    if task is None:
                        return self._send(HTTPStatus.OK, json.dumps({"task": None, **server.status()}).encode())
                    return self._send(HTTPStatus.OK, json.dumps({"task": task.public(), **server.status()}).encode())
                if url.path == "/api/status":
                    return self._send(HTTPStatus.OK, json.dumps(server.status()).encode())
                return self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}')

            def do_POST(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if not self._authorized(parse_qs(url.query)):
                    return self._send(HTTPStatus.FORBIDDEN, b'{"error": "bad token"}')
                if url.path != "/api/submit":
                    return self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}')
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    rater, served_to = (str(payload[k])[:80] if payload.get(k) else None for k in ("rater", "served_to"))
                    server.submit(payload["task_id"], payload.get("result") or {}, rater, served_to=served_to)
                except RatingConflict as e:  # the page tells the rater and moves on
                    return self._send(HTTPStatus.CONFLICT, json.dumps({"error": str(e), **server.status()}).encode())
                except (KeyError, ValueError) as e:
                    return self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(e)}).encode())
                return self._send(HTTPStatus.OK, json.dumps(server.status()).encode())

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.host, self.port = host, self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{host}:{self.port}/" + (f"?token={self.token}" if self.token else "")

    # -------------------------------------------------------------------------- queue
    def add(self, task: RatingTask) -> None:
        with self._lock:
            self._tasks[task.id] = task
            self._order.append(task.id)

    def next_task(self, rater: str) -> RatingTask | None:
        """The rater's open task if any, else the oldest unassigned (or abandoned) task."""
        now = time.time()
        with self._lock:
            for tid in self._order:
                t = self._tasks[tid]
                if t.result is None and t.rater == rater:
                    return t
            for tid in self._order:
                t = self._tasks[tid]
                stale = t.opened_at is not None and self.reassign_after_s is not None and now - t.opened_at > self.reassign_after_s
                if t.result is None and (t.rater is None or stale):
                    t.rater, t.opened_at = rater, now
                    return t
        return None

    def submit(self, task_id: str, result: dict[str, Any], rater: str | None = None, *,
               served_to: str | None = None) -> None:
        """Record a judgment by ``rater``; ``served_to`` is who the task was handed to, if the rater has
        renamed since. Raises :class:`RatingConflict` rather than silently dropping a judgment that cannot
        count: a second answer to a task, an answer by someone the task is not assigned to (two tabs, a task
        reassigned after ``reassign_after_s``), or one for a cancelled task."""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                raise KeyError(f"unknown task {task_id}")
            if t.cancelled:
                raise RatingConflict("no longer needed")
            if t.result is not None:
                raise RatingConflict("already answered")
            holder = served_to or rater
            if holder and t.rater and holder != t.rater:
                raise RatingConflict("taken by another rater")
            now = time.time()
            elapsed = now - (t.opened_at or t.created_at)
            auto = bool(result.get("auto"))  # submitted by the countdown: over budget by definition
            t.result = {**result, "auto": auto, "elapsed_s": elapsed, "rater": rater or t.rater,
                        "over_budget": auto or bool(t.budget_s and elapsed > t.budget_s + 2)}
            self._order.remove(task_id)
            self._done += 1
        if self.log_path is not None:
            rec = {"task_id": t.id, "title": t.title, "kind": t.kind, "budget_s": t.budget_s, **t.result,
                   "submitted_at": now}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if t.loop is not None and t.future is not None:
            try:
                t.loop.call_soon_threadsafe(lambda: t.future.done() or t.future.set_result(t.result))
            except RuntimeError:  # the episode's event loop is gone (run aborted); the judgment stays logged
                pass

    def cancel(self, task_id: str) -> None:
        """Withdraw an unanswered task whose decision nobody awaits any more (e.g. the run was interrupted),
        so that raters are not served tasks of dead episodes; a late answer to it gets HTTP 409."""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is not None and t.result is None and not t.cancelled:
                t.cancelled = True
                self._order.remove(task_id)

    def status(self) -> dict[str, int]:
        with self._lock:
            waiting = sum(1 for tid in self._order if self._tasks[tid].rater is None)
            return {"waiting": waiting, "in_progress": len(self._order) - waiting, "done": self._done}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class HumanPolicy(Policy):
    """A person, via a :class:`RatingServer`: each decision is a rating task with the request's form.

    Args:
        time_budget_s: evaluation budget shown as a countdown (the experimental variable).
        enforce_budget: submit what the rater has entered when the budget runs out. What they left
            unanswered is an abstention (``parse_ok=False``, flagged ``auto`` and ``over_budget``), never a
            default answer such as the first option, which would bias budget curves.
        show_prompt: always unfold the role's instructions (its system messages). They are always shown - a
            private brief, a side objective or the reward rule is part of the role's game - and by default
            long ones are folded once the rater has seen them.
    """

    def __init__(self, server: RatingServer, *, time_budget_s: float | None = None, enforce_budget: bool = False,
                 show_prompt: bool = False, label: str | None = None, id: str | None = None):
        super().__init__(label=label or (f"human-{int(time_budget_s)}s" if time_budget_s else "human"), id=id)
        self.server, self.time_budget_s, self.enforce_budget, self.show_prompt = server, time_budget_s, enforce_budget, show_prompt

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "time_budget_s": self.time_budget_s, "enforce_budget": self.enforce_budget}

    async def act(self, request: ActionRequest, ctx: ActContext) -> Action:
        msgs = list(request.prompt)  # system messages included: a person must know what the role's model is told
        game = ctx.game
        title = f"{game.mechanism.name} · {ctx.role}" + (f" · {request.phase}" if request.phase else "") if game else ctx.role
        if request.kind == "text":
            hint = format_instructions(request)
            msgs = [*msgs, type(msgs[0])(role="user", content=hint)] if hint and msgs else msgs
        options = list(request.options or [])
        texts = dict(request.option_texts or {})
        if game is not None and game.item.answers:  # show "(A) 242000", not just "A"
            texts = {**{a.label: a.render() for a in game.item.answers if a.label in options}, **texts}
        loop = asyncio.get_running_loop()
        task = RatingTask(id=uuid.uuid4().hex[:12], kind=request.kind, title=title,
                          messages=[{"role": m.role, "content": m.content} for m in msgs],
                          options=options, option_texts=texts,
                          score_range=request.score_range, score_meaning=request.score_meaning,
                          budget_s=self.time_budget_s, enforce_budget=self.enforce_budget, loop=loop,
                          future=loop.create_future(), expand_instructions=self.show_prompt,
                          json_keys=list(request.json_keys or []) if request.kind == "json" else [])
        self.server.add(task)
        try:
            res: dict[str, Any] = await task.future
        except BaseException:  # the run was interrupted: raters must not be served a task nobody waits for
            self.server.cancel(task.id)
            raise
        action = human_action(request, res)
        action.usage = Usage(calls=1, effort_seconds=float(res.get("elapsed_s", 0.0)))
        auto = bool(res.get("auto"))
        action.metadata.update({"human": {"rater": res.get("rater"), "elapsed_s": res.get("elapsed_s"),
                                          "budget_s": self.time_budget_s, "auto": auto,
                                          "over_budget": auto or bool(res.get("over_budget"))}})
        return action


def human_action(request: ActionRequest, res: dict[str, Any]) -> Action:
    """The action for a rater's answer ``res`` (as the page posts it) to ``request``.

    What the rater left unanswered - the countdown submitted before they chose - is an abstention with
    ``parse_ok=False`` (no choice or score, uniform probabilities, no data): mechanisms apply their own
    fallback, and analyses can tell it from a real answer. JSON answers come one field per key, each read
    as JSON when it parses (``true``, ``null``, ``3``) and as text otherwise; empty fields are omitted.
    """
    auto = bool(res.get("auto"))
    note = str(res.get("rationale") or "").strip()
    opts = request.options or []
    if request.kind == "probabilities":
        probs = res.get("probs")
        action = coerce_action(request, probs) if isinstance(probs, dict) and probs else neutral_action(request)
        action.text = note or (action.text if action.parse_ok else "")
    elif request.kind == "choice":
        c = res.get("choice")
        ok = c is not None and (c in opts or not opts)
        action = Action(text=note or str(c), choice=c) if ok else Action(text=note, parse_ok=False)
    elif request.kind == "score":
        try:
            s = float(res["score"])
        except (KeyError, TypeError, ValueError):
            s = None
        action = Action(text=note or f"{s:g}", score=s) if s is not None else Action(text=note, parse_ok=False)
    elif request.kind == "json" and isinstance(res.get("data"), dict):
        data = {str(k): _field_value(v) for k, v in res["data"].items() if not (isinstance(v, str) and not v.strip())}
        action = Action(text=note or json.dumps(data), data=data) if data or not auto else neutral_action(request, note)
    elif request.kind == "json":  # a JSON object typed as text (no keys given): read like a model's reply
        action = finalize_from_text(request, str(res.get("text") or ""))
        action.text = note or action.text
    else:
        text = str(res.get("text") or "")
        action = Action(text=text, parse_ok=bool(text.strip()) or not auto)
    return action


def _field_value(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v.strip()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judge</title>
<style>
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--warn:#e34948;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--warn:#e66767;--chip:#2c2c2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);z-index:2}
.bar{height:3px;background:var(--accent);width:100%;transition:width 1s linear}.bar.low{background:var(--warn)}
.top{display:flex;gap:12px;align-items:center;padding:8px 16px;font-size:13px;color:var(--ink2)}
.top .grow{flex:1}.top input{font:inherit;padding:2px 6px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--ink);width:9em}
.time{font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}.time.low{color:var(--warn)}
main{max-width:860px;margin:0 auto;padding:16px}
.msg{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.msg .who{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.msg .body{white-space:pre-wrap;word-wrap:break-word}
pre{background:var(--chip);padding:8px;border-radius:6px;overflow:auto;white-space:pre-wrap;margin:6px 0}
.answer{position:sticky;bottom:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:14px;box-shadow:0 -4px 12px rgba(0,0,0,.04)}
.opt{display:grid;grid-template-columns:minmax(80px,1fr) 3fr 3.5em;gap:10px;align-items:center;margin:4px 0}
.opt label,.kv label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.opt output{text-align:right;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:minmax(80px,1fr) 3fr;gap:10px;align-items:center;margin:4px 0}
input[type=range]{width:100%;accent-color:var(--accent)}
.row{display:flex;gap:10px;align-items:center;margin-top:10px}.row .grow{flex:1}
textarea,.kv input{width:100%;font:inherit;border:1px solid var(--line);border-radius:8px;padding:8px;background:transparent;color:var(--ink)}
textarea{min-height:4em}.kv input{padding:5px 8px}
button{font:inherit;font-weight:600;border:0;border-radius:8px;padding:8px 18px;background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.choice label{display:block;padding:4px 0}.empty{text-align:center;color:var(--ink2);padding:60px 0}.empty button{margin-top:12px}
details.msg:not([open]) .who{margin-bottom:0}
.toast{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);background:var(--ink);color:var(--card);padding:6px 12px;
border-radius:8px;font-size:13px;z-index:9;pointer-events:none}
.h{font-weight:600;margin-top:.6em}
.claim{border-radius:6px;padding:1px 6px;border:1px solid var(--line);white-space:normal}
.claim code{font-size:13px}.claim.verified{border-color:#0ca30c}.claim.failed{border-color:var(--warn)}
.claim .mark{font-weight:600}.claim.verified .mark{color:#0ca30c}.claim.failed .mark{color:var(--warn)}.claim.unverified .mark,.claim.executed .mark{color:var(--muted)}
[data-tip]{position:relative;cursor:help;border-bottom:1px dotted var(--muted)}
[data-tip]:hover::after,[data-tip]:focus::after{content:attr(data-tip);position:absolute;left:0;top:1.6em;width:260px;background:var(--ink);color:var(--card);
padding:6px 8px;border-radius:6px;font-size:12px;font-weight:400;z-index:5;white-space:normal}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}
@media (max-width:600px){#title{display:none}.top{gap:8px}main{padding:10px}.answer{position:static}
.opt{grid-template-columns:1fr 3.5em}.opt label{grid-column:1/-1;white-space:normal}.kv{grid-template-columns:1fr;gap:2px}}
</style></head><body>
<header><div class="bar" id="bar" style="width:0"></div>
<div class="top"><span id="title">Waiting for judgments</span><span class="grow"></span>
<span id="queue" data-tip="Judgments waiting / done" tabindex="0"></span>
<span class="time" id="time"></span>
<input id="rater" placeholder="your name" aria-label="Your name" title="Recorded with your judgments; leave empty to stay anonymous"></div></header>
<main id="main"><div class="empty">No judgment waiting. This page checks for new ones automatically.</div></main>
<div class="toast" id="toast" role="status" hidden></div>
<script>
const token = new URLSearchParams(location.search).get("token") || "";
const q = (p) => p + (p.includes("?") ? "&" : "?") + (token ? "token=" + encodeURIComponent(token) : "");
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const rater = $("rater"); rater.value = localStorage.getItem("rater") || ""; rater.onchange = () => localStorage.setItem("rater", rater.value);
// raters without a name still need distinct ids: the server gives each rater their own task
const anon = localStorage.getItem("rater_id") || "anon-" + [...crypto.getRandomValues(new Uint8Array(4))].map(b => b.toString(16).padStart(2, "0")).join("");
localStorage.setItem("rater_id", anon);
const who = () => rater.value.trim() || anon;
// served: who the shown task was handed to (the rater may type a name meanwhile); moved: sliders touched;
// paused: after a timed-out judgment, so that an unattended tab does not time out the whole queue
let task = null, timer = null, deadline = null, sent = false, served = "", moved = false, paused = false;
const MARKS = {verified: ["\u2713 verified", "A trusted checker confirmed this claim"],
  failed: ["\u2717 failed", "A trusted checker found this claim false"],
  unverified: ["? unchecked", "This claim was not checked"],
  executed: ["\u25b8 ran", "A trusted tool ran this but checked no stated result: the output is the participant's own"]};
function body(text){ // plain text, with code blocks, headings and verified-claim markers made readable
  return esc(text)
    .replace(/```[a-z0-9]*\n?([\s\S]*?)```/gi, (m, c) => "<pre>" + c + "</pre>")
    .replace(/^### (.*)\n?/gm, '<div class="h">$1</div>')
    .replace(/&lt;(verified|failed|unverified|executed) kind=&quot;([^&]*)&quot;&gt;([\s\S]*?)&lt;\/\1&gt;/g, (m, tag, kind, inner) => {
      let res = "";
      inner = inner.replace(/&lt;result&gt;([\s\S]*?)&lt;\/result&gt;/, (mm, r) => { res = r; return ""; });
      return `<span class="claim ${tag}" data-tip="${MARKS[tag][1]} (${kind})" tabindex="0"><span class="mark">${MARKS[tag][0]}</span> ` +
        `<code>${inner.trim()}</code>${res ? " \u2192 " + res.trim() : ""}</span>`;
    });
}
function status(s){ $("queue").textContent = `${s.waiting + s.in_progress} waiting · ${s.done} done`; }
function answerForm(t){
  if (t.kind === "probabilities") {
    const rows = t.options.map((o, i) => `<div class="opt"><label title="${esc(t.option_texts[o] || o)}">${esc(t.option_texts[o] || o)}</label>
      <input type="range" min="0" max="100" value="${Math.round(100 / t.options.length)}" data-opt="${esc(o)}" aria-label="${esc(o)}">
      <output>0%</output></div>`).join("");
    return `<div><span data-tip="How likely is each answer? The sliders are rescaled to sum to 100% when you submit." tabindex="0">Your probabilities</span></div>${rows}`;
  }
  if (t.kind === "choice") return '<div class="choice">' + t.options.map(o => `<label><input type="radio" name="c" value="${esc(o)}"> ${esc(t.option_texts[o] || o)}</label>`).join("") + "</div>";
  if (t.kind === "score") { const [lo, hi] = t.score_range || [0, 10];
    return `<div class="opt"><label>${esc(t.score_meaning || "Score")}</label><input type="range" id="score" min="${lo}" max="${hi}" step="0.5" value="${(lo + hi) / 2}"><output id="scoreout"></output></div>`; }
  if (t.kind === "json" && t.json_keys.length)
    return `<div><span data-tip="Each field is read as JSON if it parses (true, false, null, numbers), otherwise as text. Empty fields are left out." tabindex="0">Your answer</span></div>` +
      t.json_keys.map((k, i) => `<div class="kv"><label for="k${i}" title="${esc(k)}">${esc(k)}</label><input id="k${i}" data-key="${esc(k)}" autocomplete="off"></div>`).join("");
  return `<textarea id="text" placeholder="${t.kind === "json" ? "A JSON object" : "Your response"}"></textarea>`;
}
function normalized(){
  const s = [...document.querySelectorAll("[data-opt]")]; const tot = s.reduce((a, x) => a + +x.value, 0);
  const out = {}; s.forEach(x => out[x.dataset.opt] = tot ? +x.value / tot : 1 / s.length); return out;
}
function refreshOutputs(){
  const p = normalized(); document.querySelectorAll("[data-opt]").forEach(x => x.nextElementSibling.textContent = Math.round(100 * p[x.dataset.opt]) + "%");
  const sc = $("score"); if (sc) $("scoreout").textContent = sc.value;
}
function fresh(text){ // first time this browser shows these instructions?
  const h = String([...text].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261) >>> 0);
  const seen = JSON.parse(localStorage.getItem("seen") || "[]");
  if (seen.includes(h)) return false;
  localStorage.setItem("seen", JSON.stringify([...seen, h].slice(-200))); return true;
}
function show(t){
  task = t; sent = false; moved = false;
  $("title").textContent = t.title;
  // the role's instructions (system messages) come first: unfolded when short or new, folded once familiar
  const sys = t.messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
  const open = sys && (t.expand_instructions || sys.length <= 400 || fresh(sys));
  const inst = sys ? `<details class="msg"${open ? " open" : ""}><summary class="who" title="What this role is told">Instructions</summary><div class="body">${body(sys)}</div></details>` : "";
  const msgs = t.messages.filter(m => m.role !== "system").map(m => `<div class="msg"><div class="who">${esc(m.role === "user" ? "task" : m.role)}</div><div class="body">${body(m.content)}</div></div>`).join("");
  $("main").innerHTML = inst + msgs + `<div class="answer">${answerForm(t)}
    <details><summary>Add a note (optional)</summary><textarea id="why" placeholder="Why?"></textarea></details>
    <div class="row"><span class="grow"></span><button id="go">Submit</button></div></div>`;
  document.querySelectorAll("input[type=range]").forEach(x => x.oninput = () => { moved = true; refreshOutputs(); }); refreshOutputs();
  $("go").onclick = () => submit(false);
  clearInterval(timer); deadline = t.budget_s ? (t.opened_at * 1000 + t.budget_s * 1000) : null; tick(); timer = setInterval(tick, 500);
  window.scrollTo(0, 0);
}
function toast(text){ const n = $("toast"); n.textContent = text; n.hidden = false; clearTimeout(n.t); n.t = setTimeout(() => n.hidden = true, 6000); }
function pause(){
  paused = true; $("title").textContent = "Paused"; $("time").textContent = ""; $("bar").style.width = "0";
  $("main").innerHTML = '<div class="empty"><div>Time is up</div><button id="resume" title="Your answer was submitted as it stood. Continue when you are ready for the next judgment.">Continue</button></div>';
  $("resume").onclick = () => { if (paused) { paused = false; poll(); } }; $("resume").focus();
}
function tick(){
  if (!task || !deadline) { $("time").textContent = ""; $("bar").style.width = "0"; return; }
  const left = Math.max(0, deadline - Date.now()) / 1000, frac = left / task.budget_s;
  $("time").textContent = `${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
  $("time").classList.toggle("low", frac < 0.2); $("bar").classList.toggle("low", frac < 0.2); $("bar").style.width = (100 * frac) + "%";
  if (left <= 0 && task.enforce_budget && !sent) submit(true);
}
async function submit(auto){
  if (!task || sent) return;
  // the countdown sends only what the rater entered: untouched sliders or no pick are an abstention, not an answer
  const result = {rationale: ($("why") || {}).value || "", auto};
  if (task.kind === "probabilities") { if (moved || !auto) result.probs = normalized(); }
  else if (task.kind === "choice") { const c = document.querySelector("input[name=c]:checked"); if (!c && !auto) return alert("Pick an answer"); if (c) result.choice = c.value; }
  else if (task.kind === "score") { if (moved || !auto) result.score = +$("score").value; }
  else if ($("text")) result.text = $("text").value;
  else { result.data = {}; document.querySelectorAll("[data-key]").forEach(x => result.data[x.dataset.key] = x.value); }
  sent = true; $("go").disabled = true;
  let r;
  try {
    r = await fetch(q("/api/submit"), {method: "POST", headers: {"Content-Type": "application/json", "X-Token": token},
      body: JSON.stringify({task_id: task.id, rater: who(), served_to: served, result})});
  } catch (e) { sent = false; $("go").disabled = false; return toast("Not sent: server unreachable"); }
  const e = await r.json().catch(() => ({}));
  if (e.done !== undefined) status(e);  // the paused view keeps an up-to-date queue count
  if (!r.ok) toast("Not recorded: " + (e.error || r.status));
  task = null; clearInterval(timer);
  if (auto) pause(); else poll();
}
async function poll(){
  if (task || paused) return;
  try {
    const me = who(), r = await (await fetch(q("/api/next?rater=" + encodeURIComponent(me)))).json();
    status(r);
    if (r.task) { served = me; return show(r.task); }
    $("title").textContent = "Waiting for judgments"; $("time").textContent = ""; $("bar").style.width = "0";
    $("main").innerHTML = '<div class="empty">No judgment waiting. This page checks for new ones automatically.</div>';
  } catch (e) {}
  setTimeout(poll, 2000);
}
poll();
</script></body></html>
"""
