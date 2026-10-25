import math
from so_arena.samplers.prompt_search import Candidate, SearchResult, PromptSearchSuite
def res(d, rewards, items):
    c=Candidate(id=d, strategy=d, rewards=rewards, item_ids=items, values=[None]*len(items))
    return SearchResult(role="agent", mechanism="m", directive=d, algorithm="opro", candidates=[c])
items=["i1","i2","i3","i4"]
s=PromptSearchSuite.__new__(PromptSearchSuite)
# the deceptive strategy's episode on i4 lost its reward (NaN) - e.g. the judge output could not be scored
s.results={"honest":res("honest",[0.0,0.0,0.0,-5.0],items),
           "deceptive":res("deceptive",[0.5,0.5,0.5,math.nan],items)}
print(s.honesty_margin())
