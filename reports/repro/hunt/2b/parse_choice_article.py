# parse_choice matches one-letter labels case-insensitively after "answer/choice/decision/verdict (is)":
# the English article "a" becomes option A.
from so_arena.core.parsing import parse_choice, mentioned_options
for t in ("The answer is a tricky one, but I pick B.",
          "My decision: a clear B.",
          "Final verdict is a narrow win for B"):
    print(f"{t!r:48} -> parse_choice={parse_choice(t, ['A', 'B'])}  mentioned_options={mentioned_options(t, ['A', 'B'])}")
