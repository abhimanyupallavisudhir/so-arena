import oversight_arena as oa
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Propaganda
from oversight_arena.elicitation import reward_function
from oversight_arena.agents.scripted import ScriptedAgent

dom = HiddenBits(n_tasks=4)
t = dom.tasks()[0]
print("options", [(o.id, o.text, o.value) for o in t.options])
seen = []
def judge(obs):
    seen.append(obs.transcript_text[-200:])
    # judge believes whatever option the advocate text names
    txt = obs.transcript_text
    for o in obs.task.options:
        if f"ANSWER={o.id}" in txt:
            return {"probs": {x.id: (0.9 if x.id == o.id else 0.1) for x in obs.task.options}}
    return {"probs": {x.id: 0.5 for x in obs.task.options}}
fn = reward_function(dom, Propaganda(), "agent", fixtures={"judge": ScriptedAgent(judge)})
ids = [o.id for o in t.options]
# two completions: one convincingly argues for option ids[0], one for ids[1]
print(fn(prompts=[None, None], completions=[f"ANSWER={ids[0]}", f"ANSWER={ids[1]}"], task_id=[t.id, t.id]))
print("with position column:", fn(prompts=[None, None], completions=[f"ANSWER={ids[0]}", f"ANSWER={ids[1]}"], task_id=[t.id, t.id], position=[ids[0], ids[1]]))
cor = next(o.id for o in t.options if o.value > 0)
st = ["correct" if i == cor else "incorrect" for i in ids]
print("with stance column:", st, fn(prompts=[None, None], completions=[f"ANSWER={ids[0]}", f"ANSWER={ids[1]}"], task_id=[t.id, t.id], stance=st))
print("stance column as a tuple of numpy strings:", fn(prompts=[None, None], completions=[f"ANSWER={ids[0]}", f"ANSWER={ids[1]}"], task_id=[t.id, t.id], stance=__import__('numpy').array(st)))
