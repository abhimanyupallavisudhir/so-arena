# ManifoldForecasting shows "Question closes: <closeTime>" to every role. For resolved markets
# closeTime is typically moved to the (early) resolution time, so "closes well before the date in
# the question / before the scheduled close" correlates with the outcome. Measure on live data:
# accuracy of the blind rule "closed early (close within 1 day of resolution AND > 30 days before
# the market's ... )" is not needed; we just check how often close == resolution time by outcome.
import json, urllib.request
from oversight_arena.domains.forecasting import ManifoldForecasting, _iso
import os
os.environ.setdefault("OA_DATA_DIR", "/tmp/hunt/1b/oa_data")
ms = ManifoldForecasting(n=300)._search()
rows = []
for m in ms:
    c, r = m.get("closeTime"), m.get("resolutionTime")
    if not c or not r:
        continue
    rows.append((m["resolution"], abs(c - r) < 86400e3, (m.get("closeTime") - m["createdTime"]) / 86400e3))
for res in ("YES", "NO"):
    sub = [x for x in rows if x[0] == res]
    print(f"{res}: n={len(sub)} close within 1 day of resolution: {sum(x[1] for x in sub)/max(1,len(sub)):.2f}")
t = ManifoldForecasting(n=5).tasks()[0]
print("shown to agents:", [b.content for b in t.view().info if b.key == "meta"], "| metadata:", t.view().metadata)
