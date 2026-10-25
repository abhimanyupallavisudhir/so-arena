"""Human participants through a local web page (e.g. human judges for debate or consultancy).

    judge = WebHumanAgent(port=8770)          # open http://127.0.0.1:8770 in a browser
    Experiment(domain, Debate(), {"kind:judge": judge, "*": oa.llm_agents()["*"]}, Stances()).run()

Each time the agent is asked to act, the request appears on the page: task, visible transcript,
private instructions (if any) and a response form matching what the mechanism expects
(probabilities, a choice, a number or text). Several concurrent episodes queue up; answers are
returned to the episodes that asked. The page shows only what the role may see.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .base import Action, Agent, Observation

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OversightArena</title><style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#34342f;--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:980px;margin:0 auto;padding:20px}.mut{color:var(--mut)}h1{font-size:18px;margin:0}
.box{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0;white-space:pre-wrap}
.row{display:flex;gap:10px;align-items:center;margin:6px 0}.row label{min-width:5em;font-weight:600}
input[type=range]{flex:1}button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:8px 16px;font-size:15px;cursor:pointer}
button.opt{background:transparent;color:var(--fg);border:1px solid var(--line)}textarea,input[type=number]{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--fg)}
details summary{cursor:pointer;color:var(--mut)}
mark{border-radius:4px;padding:0 3px}mark.VERIFIED{background:#d7f0e1;color:#0b5d2e}mark.REFUTED{background:#fbdcdc;color:#8a1c1c}
mark.UNCHECKED,mark.CHECKED{background:rgba(127,127,127,.15);color:inherit}
</style></head><body><main>
<div class="row" style="justify-content:space-between"><h1 id="title">Waiting for the next turn…</h1><span class="mut" id="queue"></span></div>
<div id="req"></div></main><script>
let cur=null;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]);
const marks={VERIFIED:'✓ verified',REFUTED:'✗ refuted',UNCHECKED:'unchecked',CHECKED:'checked'};
const pretty=s=>esc(s).replace(/&lt;(\w+)((?:\s+\w+="[^"]*")*?)\s+status="(\w+)"((?:\s+\w+="[^"]*")*)&gt;([\s\S]*?)&lt;\/\1&gt;/g,
 (m,tag,a1,st,a2,body)=>`<mark class="${st}" title="${marks[st]||st}">${esc(tag)}${(a1+a2).replace(/\s+(\w+)="([^"]*)"/g,' $1=$2')}: ${body}</mark>`);
async function poll(){try{const r=await (await fetch('api/pending')).json();document.getElementById('queue').textContent=r.length>1?(r.length-1)+' more waiting':'';
 if(!r.length){cur=null;document.getElementById('title').textContent='Waiting for the next turn…';document.getElementById('req').innerHTML='';}
 else if(!cur||cur.id!==r[0].id){cur=r[0];show(cur)}}catch(e){} setTimeout(poll,1000)}
function show(q){document.getElementById('title').textContent=q.role_title+' · '+q.mechanism;
 let h='';
 if(q.brief)h+=`<details><summary>Your role</summary><div class="box">${esc(q.brief)}</div></details>`;
 if(q.strategy)h+=`<div class="box"><b>Private instructions</b>\n${esc(q.strategy)}</div>`;
 h+=`<div class="box">${esc(q.task_text)}</div>`;
 if(q.transcript_text)h+=`<div class="box">${pretty(q.transcript_text)}</div>`;
 h+=`<div class="box"><b>${esc(q.prompt||'Your turn')}</b>${q.claim_help?'\n'+esc(q.claim_help):''}</div>`;
 const r=q.response,o=r.options||[],t=r.option_texts||{};
 if(r.kind==='distribution'){h+=o.map(k=>`<div class="row"><label title="${esc(t[k]||'')}">${esc(k)}</label><input type="range" min="0" max="100" value="${Math.round(100/o.length)}" data-k="${esc(k)}" oninput="norm()"><span data-v="${esc(k)}"></span></div>`).join('')+'<button onclick="send()">Submit</button>';}
 else if(r.kind==='choice'){h+='<div class="row">'+o.map(k=>`<button class="opt" title="${esc(t[k]||'')}" onclick="send('${esc(k)}')">${esc(k)}${t[k]?' · '+esc(t[k]).slice(0,40):''}</button>`).join('')+'</div>';}
 else if(r.kind==='scalar'){h+=`<div class="row"><label>${esc(r.scalar_name)}</label><input type="number" id="num" min="${r.lo}" max="${r.hi}" step="any"></div><button onclick="send()">Submit</button>`;}
 else{h+=`<textarea id="txt" rows="6" placeholder="${r.kind==='json'?esc(JSON.stringify(r.fields)):''}"></textarea><div class="row"><button onclick="send()">Submit</button></div>`;}
 document.getElementById('req').innerHTML=h;norm()}
function norm(){const s=[...document.querySelectorAll('input[type=range]')];const tot=s.reduce((a,x)=>a+ +x.value,0)||1;
 s.forEach(x=>{document.querySelector(`[data-v="${x.dataset.k}"]`).textContent=Math.round(100*x.value/tot)+'%'})}
async function send(choice){const r=cur.response;let body={id:cur.id};
 if(r.kind==='distribution'){const s=[...document.querySelectorAll('input[type=range]')];const tot=s.reduce((a,x)=>a+ +x.value,0)||1;body.probs=Object.fromEntries(s.map(x=>[x.dataset.k,x.value/tot]))}
 else if(r.kind==='choice'){body.choice=choice}else if(r.kind==='scalar'){body.value=+document.getElementById('num').value}else{body.text=document.getElementById('txt').value}
 await fetch('api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});cur=null;document.getElementById('req').innerHTML='';}
poll();
</script></body></html>"""


class _Server:
    _instances: dict[tuple[str, int], "_Server"] = {}
    _lock = threading.Lock()

    def __init__(self, host: str, port: int):
        self.pending: dict[int, tuple[dict[str, Any], asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self.ids = itertools.count(1)
        self.mu = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # quiet
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/api/pending"):
                    with server.mu:
                        items = [dict(q, id=i) for i, (q, _, _) in sorted(server.pending.items())]
                    self._send(200, json.dumps(items).encode(), "application/json")
                else:
                    self._send(200, _PAGE.encode(), "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/api/answer"):
                    self._send(404, b"", "text/plain")
                    return
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with server.mu:
                    item = server.pending.pop(int(data.get("id", -1)), None)
                if item is None:
                    self._send(404, b"unknown request", "text/plain")
                    return
                _, loop, fut = item
                loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(data))
                self._send(200, b"ok", "text/plain")

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @classmethod
    def get(cls, host: str, port: int) -> "_Server":
        with cls._lock:
            if (host, port) not in cls._instances:
                cls._instances[(host, port)] = _Server(host, port)
            return cls._instances[(host, port)]

    def submit(self, payload: dict[str, Any]) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self.mu:
            self.pending[next(self.ids)] = (payload, loop, fut)
        return fut

    def close(self) -> None:
        self.httpd.shutdown()


class WebHumanAgent(Agent):
    """A human playing a role through a local web page (see module docstring)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8770, id: str = "web_human"):
        self.host, self.port, self.id = host, port, id

    @property
    def url(self) -> str:
        return _Server.get(self.host, self.port).url

    def describe(self) -> dict[str, Any]:
        return {"type": "WebHumanAgent", "id": self.id}

    async def act(self, obs: Observation) -> Action:
        server = _Server.get(self.host, self.port)
        payload = {
            "role": obs.role, "role_title": obs.role_title, "mechanism": obs.mechanism, "brief": obs.brief,
            "strategy": obs.strategy, "task_text": obs.task_text, "transcript_text": obs.transcript_text,
            "prompt": obs.prompt, "claim_help": obs.claim_help, "response": obs.response.model_dump(mode="json"),
        }
        data = await server.submit(payload)
        spec = obs.response
        if spec.kind == "distribution":
            probs = {k: float(v) for k, v in (data.get("probs") or {}).items()}
            return Action(text=json.dumps(probs), parsed={"probs": probs})
        if spec.kind == "choice":
            return Action(text=f"ANSWER: {data.get('choice')}", parsed={"choice": data.get("choice")})
        if spec.kind == "scalar":
            v = data.get("value")
            return Action(text=str(v), parsed={spec.scalar_name: None if v is None else float(v)})
        text = str(data.get("text", ""))
        if spec.kind == "json":
            from .parsing import parse_json

            return Action(text=text, parsed=parse_json(text) or {})
        return Action(text=text)


__all__ = ["WebHumanAgent"]
