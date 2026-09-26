# Post-resolution notes that slip through clean_description(resolved=True) (they reach agents in the question).
from so_arena.domains.forecasting import clean_description
for p in ("Resolves YES. Biden withdrew on July 21.",
          "Resolves NO, he did not attend.",
          "This happened on March 3 - YES.",
          "Closing note: the event occurred, so YES.",
          "Resolved YES because it happened."):
    out, n = clean_description("Will X happen?\n\n" + p, resolved=True)
    print(f"{p!r:45} kept={p in out}")
