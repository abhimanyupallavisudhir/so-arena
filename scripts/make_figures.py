"""Regenerate the README figures and the demo reports from the offline demos: python scripts/make_figures.py"""

import shutil
import tempfile
from pathlib import Path

from so_arena import demos

DOCS = Path(__file__).resolve().parent.parent / "docs"
DEMOS = {"asd": demos.demo_asd, "optimization": demos.demo_optimization, "swarm": demos.demo_swarm,
         "work": demos.demo_work}

if __name__ == "__main__":
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in DEMOS.items():
            run = fn(Path(tmp) / name)
            for png in (run / "figures").glob("*.png"):
                shutil.copy(png, DOCS / "figures" / png.name)
                print(DOCS / "figures" / png.name)
            shutil.copy(run / "report.html", DOCS / "reports" / f"{name}.html")
            print(DOCS / "reports" / f"{name}.html")
