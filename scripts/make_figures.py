"""Regenerate the README figures and the demo reports from the offline demos.

    python scripts/make_figures.py            # every demo
    python scripts/make_figures.py swarm release
"""

import shutil
import sys
import tempfile
from pathlib import Path

from so_arena import demos

DOCS = Path(__file__).resolve().parent.parent / "docs"
DEMOS = {"asd": demos.demo_asd, "optimization": demos.demo_optimization, "swarm": demos.demo_swarm,
         "work": demos.demo_work, "monitoring": demos.demo_monitoring, "hiddenbits": demos.demo_hiddenbits,
         "bon_budget": demos.demo_bon_budget, "release": demos.demo_release}

if __name__ == "__main__":
    names = sys.argv[1:] or list(DEMOS)
    unknown = [n for n in names if n not in DEMOS]
    if unknown:
        sys.exit(f"unknown demos {unknown}; choose from {list(DEMOS)}")
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            run = DEMOS[name](Path(tmp) / name)
            for png in (run / "figures").glob("*.png"):
                shutil.copy(png, DOCS / "figures" / png.name)
                print(DOCS / "figures" / png.name)
            shutil.copy(run / "report.html", DOCS / "reports" / f"{name}.html")
            print(DOCS / "reports" / f"{name}.html")
