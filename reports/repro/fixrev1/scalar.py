from oversight_arena.agents.parsing import parse_scalar
cases = [("SUSPICION: 7 out of 10", 0, 10, "suspicion"), ("CONFIDENCE: 8 out of 10", 0, 1, "probability"),
         ("PROBABILITY: 1.5", 0, 1, "probability"), ("My suspicion is 8/10.", 0, 10, "suspicion"),
         ("My suspicion is 3/10 given 10 files", 0, 10, "suspicion"), ('{"suspicion": "3/10"}', 0, 10, "suspicion"),
         ('{"probability": 0.7}', 0, 1, "probability"), ('{"score": 1.2}', 0, 1, "score"),
         ("SCORE: 7/10", 0, 1, "score"), ("SCORE: -3", 0, 10, "score"), ("SCORE: 1e-3", 0, 1, "score")]
for t, lo, hi, n in cases: print(f"{t!r:45} [{lo},{hi}] -> {parse_scalar(t, lo, hi, n)}")
