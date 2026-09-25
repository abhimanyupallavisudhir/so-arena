"""Regenerate the README figures from the offline demos: python scripts/make_figures.py"""

import shutil
import tempfile
from pathlib import Path

from so_arena import demos

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (demos.demo_asd, demos.demo_optimization, demos.demo_swarm):
            run = fn(Path(tmp) / fn.__name__)
            for png in (run / "figures").glob("*.png"):
                shutil.copy(png, OUT / png.name)
                print(OUT / png.name)
