"""Rebuild the demo gallery in docs/: run every built-in demo (no API keys needed), copy the HTML
reports to docs/gallery/ and the key figures to docs/figures/, and write docs/gallery/index.html.

    python scripts/build_gallery.py                 # ~5 minutes
    python scripts/build_gallery.py --from runs/demos   # reuse existing demo outputs
"""

from __future__ import annotations

import argparse
import html
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (demo, title, finding, [(figure file, gallery name)], tooltip)
CARDS = [
    ("hiddenbits", "Verified claims", "Debate's ASD grows with the verification budget; consultancy and propaganda barely move.",
     [("asd_budget_credulous.png", "hiddenbits_asd_budget.png")],
     "HiddenBits: exact Bayesian judges, programmatic advocates that lie when true evidence runs out; ASD vs verification budget "
     "for a credulous and a rational judge, next to the exact disclosure theory."),
    ("chess", "Chess: a real capability gap", "Verified engine lines lift a trapped depth-2 judge from 0% to 90% accuracy, unless advocates cherry-pick.",
     [("accuracy_honest.png", "chess_accuracy_honest.png"), ("accuracy_cherry_pick.png", "chess_accuracy_cherry_pick.png")],
     "Lichess puzzles where a depth-2 search prefers the wrong move; Stockfish depth-14 ground truth; engine-backed advocates "
     "with honest or cherry-picked lines; an argument-style game between the debaters."),
    ("bon", "Best-of-N", "Selection makes verified consultants more honest, and teaches a debate liar to lie beyond the verification budget.",
     [("bon_honesty.png", "bon_honesty.png")],
     "One step of optimisation: exact best-of-n selection from 32 base-policy samples per task; reward vs honesty of claims."),
    ("optimize", "Strategy search", "Under optimisation debate keeps a frontier gap (0.26); propaganda's vanishes (0.001).",
     [("search_debate.png", "optimize_debate.png")],
     "Evolutionary search for the best-rewarded advocate strategy under three steerings (honest only / anything / arguing "
     "falsehood), with held-out evaluation."),
    ("swarm", "Swarms", "A shared reward makes misconduct the only equilibrium; bounties above each worker's stake leave only honesty.",
     [("basin_theory_vs_sim.png", "swarm_basin.png"), ("sampled_training_cheat.png", "swarm_training.png"),
      ("theory_regimes.png", "swarm_regimes.png")],
     "Three workers, one can tamper with the scorer; empirical games under a shared reward vs whistleblower bounties; "
     "learning dynamics vs mean-field theory; natural PG vs REINFORCE."),
    ("monitoring", "Monitoring as a reward", "Penalising a real monitor's alarms buys obfuscation; a fixed-accuracy probe deters.",
     [("misbehaviour_vs_penalty.png", "monitoring_misbehaviour.png"), ("alarms_vs_penalty.png", "monitoring_alarms.png")],
     "Actor chooses honest / overt / covert misbehaviour; reward = measured success − λ·alarm; a log-reading monitor vs "
     "simulated probes with fixed AUROC (docs/theory.md, Proposition 4)."),
    ("release", "Release now, resolve later", "A judge's released ranking rewarded overconfident forecasters; resolution favoured calibrated ones.",
     [("released_ranking.png", "release_ranking.png"), ("resolved_scores.png", "release_resolved.png")],
     "Judge-rated forecasts published (hash-committed) before questions resolve, then scored with the proper log score."),
]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OversightArena demos</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#34342f;--acc:#3987e5}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1180px;margin:0 auto;padding:24px}}h1{{font-size:20px;margin:0 0 4px}}.mut{{color:var(--mut)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:18px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.card img{{width:100%;display:block;background:#fff;border-bottom:1px solid var(--line)}}.body{{padding:10px 12px}}
.card h2{{font-size:15px;margin:0 0 4px}}.card a{{color:var(--acc);text-decoration:none}}.card a:hover{{text-decoration:underline}}
.links{{margin-top:6px;font-size:13px}}[title]{{cursor:help}}
</style></head><body><main>
<h1>OversightArena demos</h1><div class="mut">Built-in experiments that run without API keys. Hover a card for its setup.</div>
<div class="grid">{cards}</div></main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="existing demo output directory")
    args = ap.parse_args()
    src = Path(args.src) if args.src else Path(tempfile.mkdtemp(prefix="oa_gallery_"))
    if args.src is None:
        from oversight_arena import demos

        demos.run("all", src)
    gallery, figures = ROOT / "docs" / "gallery", ROOT / "docs" / "figures"
    shutil.rmtree(gallery, ignore_errors=True)
    gallery.mkdir(parents=True)
    figures.mkdir(parents=True, exist_ok=True)
    cards = []
    for demo, title, finding, figs, tip in CARDS:
        (gallery / demo).mkdir()
        shutil.copy(src / demo / "report.html", gallery / demo / "report.html")
        for f, name in figs:
            shutil.copy(src / demo / f, figures / name)
        links = [f'<a href="{demo}/report.html">report</a>']
        if demo == "release":
            shutil.copytree(src / demo / "release", gallery / demo / "release")
            links.append('<a href="release/release/index.html" title="The published release page (no ground truth), now resolved">release page</a>')
        cards.append(
            f'<div class="card" title="{html.escape(tip)}"><a href="{demo}/report.html"><img alt="{html.escape(title)}" '
            f'src="../figures/{figs[0][1]}"></a><div class="body"><h2>{html.escape(title)}</h2><div>{html.escape(finding)}</div>'
            f'<div class="links">{" · ".join(links)}</div></div></div>')
    (gallery / "index.html").write_text(PAGE.format(cards="".join(cards)))
    print(gallery / "index.html")


if __name__ == "__main__":
    main()
