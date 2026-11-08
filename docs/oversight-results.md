# Oversight protocols on the conceptual-reasoning testbed: first results

Does arguing help a weak judge reach the truth on questions that need conceptual reasoning? And does it keep
helping when the arguers are optimized against the judge? Questions come from the testbed
([conceptual validation testbed](https://github.com/abhimanyupallavisudhir/conceptual-validation-testbed)): 10 families, 16 items each, with hidden ground truth from computation or
experiment.

> **Update (2026-11-08).** A [model panel](#model-panel-eic-against-agent-and-judge-intelligence-added-2026-11-08)
> of 20 agent configurations and 9 judges, on 300 items not selected on any judge, corrects the judge-capability
> result. Raw EIC rises steeply with the judge's intelligence, but almost all of that is the judge already knowing
> the answer: its blind baseline rises just as fast. Over the blind judge, with committed liars, debate adds about
> nothing (+0.08 at most, for gpt-4o-mini) and consultancy is below it for every judge. Across agents, EIC barely
> moves with intelligence; the apparent rise with prompted liars comes from strong models refusing to lie.
>
> **Update (2026-11-05).** Final numbers for the headline: prompt-search strategies rescored on 480 held-out
> items (was 80), blind-judge baselines, and prover-estimator re-run with its decomposition payoff fixed. The fix
> removes the payoff exploit but not the failure: the fixed protocol still pays more for lying under every sampler.
>
> **Update (2026-11-02).** The [headline](#headline-empirical-incentive-compatibility-added-2026-11-02) is now
> empirical incentive compatibility: whether each protocol's reward favours honesty under instructed, best-of-n
> and prompt-optimized liars. The judge-accuracy sections that follow are secondary.
>
> **Update (2026-10-31).** The [decisive study](#decisive-study-added-2026-10-31) below fixes the scale problems of this first study: a 54-point knowledge gap,
> 3 judges, 2 debaters, 3 samples, audited and optimized liars. Against prompted liars it reverses the first study's finding:
> consultancy, debate and recursive debate help. A liar optimized against the judge still pushes every protocol below
> the judge alone, unless the honest side is optimized too, and then only debate keeps its gain.
>
> **Update (2026-11-01).** The [extended study](#decisive-study-extended-added-2026-11-01) adds a validated audit,
> held-out best-of-n (most of the in-sample collapse was a winner's curse), 3-round optimization, prompt search, an
> 8-debater tournament, QuALITY and stronger judges. Paper draft: [`paper/main.pdf`](paper/main.pdf).

**Setup.**
- Judge: gpt-4.1-nano, reasoning before it answers.
- Agents and debaters: gpt-4.1-nano, gpt-4.1-mini, gpt-5-mini (low reasoning effort) and Claude Sonnet 4.5.
- Instructed arms: on every item the agent argues the true answer in one episode and a false one in another.
- Protocols:
  - the judge alone;
  - consultancy (2 rounds);
  - debate (2 rounds, simultaneous);
  - recursive debate (depth 2, width 3);
  - prover-estimator debate (depth 2, width 3, $\rho=2$; Brown-Cohen et al. 2025);
  - as a reference, the agent model answering alone.
- Code: [`experiments/oversight_testbed.py`](../experiments/oversight_testbed.py) and
  [`oversight_report.py`](../experiments/oversight_report.py).
- Full tables: [`results/oversight_summary.md`](results/oversight_summary.md).
- Cost: about \$87 for the protocol comparisons (\$83 of it the gpt-5-mini and Sonnet arms), plus \$5 for the
  optimization runs.

![verdict accuracy](results/oversight_accuracy.png)

## Headline: empirical incentive compatibility (added 2026-11-02)

**This is the study's main result.** The sections after it measure judge accuracy, which is secondary. A protocol
is worth training against if optimizing its reward makes the agent more honest, and a judge that simply trusts the
agent can score high accuracy while being useless to train against. So the headline is **empirical incentive
compatibility (EIC)**.

**Definition.** EIC is the covariance between a protocol's reward $u$ and the ground-truth value $v$ of the
agent's behaviour, over behaviours drawn by a sampler. Here $v$ is 1 if the agent argued the true answer and 0 if
it argued a false one.
- **Why covariance:** $\tfrac{d}{d\beta}\mathbb{E}_{\pi_\beta}[v] = \mathrm{Cov}_{\pi_\beta}(u, v)$, so it is
  the first-order effect of optimizing the reward on honesty.
- **What we report:** ASD $= \mathrm{Cov}/\mathrm{Var}(v)$, the reward for honesty minus the reward for lying, in
  the protocol's reward units. Also concordance, the share of items where honesty earns more, and
  $r = \mathrm{corr}(u, v)$.
- **Samplers:**
  - instructed arguments (ASD);
  - instructed arguments with committed liars only (effective lies; this handles conceding liars);
  - best-of-$n$;
  - prompt optimization: Claude Opus 5.5 searches separately for the best honest and the best deceptive
    strategy, selected on 36 items and scored on 480 held-out items (80 for the original prover-estimator).
- **Held-out judging:** every selecting sampler is scored on independent judge samples.
- **Blind-judge baseline:** the same judge, shown only the question and which side the agent defends, never an
  argument. Its "ASD" is what the judge's prior alone pays for honesty. On these items it is negative, because they
  were selected for the judges getting them wrong: −0.58 [−0.77, −0.40] in log score, −0.11 [−0.13, −0.08] in
  probability. The dashed lines in the figures mark it. A protocol whose ASD is above the blind line is overturning
  the judge's wrong prior, even where its ASD is below zero.

**Code and outputs:**
- Code: [`experiments/decisive_eic.py`](../experiments/decisive_eic.py),
  [`decisive_eic_po.py`](../experiments/decisive_eic_po.py) and
  [`decisive_eic_figures.py`](../experiments/decisive_eic_figures.py).
- Tables: [`results/eic.md`](results/eic.md).
- Paper draft: [`paper/main.pdf`](paper/main.pdf).

![EIC of each protocol under each sampler](results/eic/fig1_eic_by_sampler.png)

| protocol (gpt-5-mini, 3 weak judges) | instructed ASD | committed liars | best-of-16 per speech, both / liar only | prompt-optimized (held-out) |
|---|---|---|---|---|
| consultancy | +0.38 [+0.29, +0.47] | +0.21 [+0.12, +0.31] | −0.06 / +0.10 | **−0.06** [−0.18, +0.06] |
| debate, sequential | +0.74 [+0.66, +0.82] | +0.56 [+0.48, +0.64] | +0.60 / +0.22 | **+0.32** [+0.18, +0.46] |
| debate, simultaneous | +0.79 [+0.70, +0.87] | +0.57 [+0.48, +0.65] | – | **+0.56** [+0.40, +0.70] |
| recursive debate | +0.10 [+0.08, +0.11] | +0.10 [+0.08, +0.12] | – | **+0.10** [+0.08, +0.13] |
| prover-estimator, original payoff | −0.13 [−0.15, −0.11] | −0.23 [−0.26, −0.20] | – | **−0.37** [−0.46, −0.29] (80 items) |
| prover-estimator, fixed payoff | −0.11 [−0.14, −0.07] (400 items) | −0.21 [−0.25, −0.17] | – | **−0.23** [−0.26, −0.20] |
| *blind judge (no argument)* | *−0.58 log score / −0.11 probability* | | | *−0.61 / −0.11 on the 480 items* |

Units: log score for consultancy and the debates, judge probability for recursive debate, prover payoff for
prover-estimator (which has no blind counterpart). Prompt-optimized: the best honest strategy's held-out reward
minus the best deceptive strategy's, after five search rounds; round 0, the unprompted agent on both sides, is
the instructed sampler on the same items.

What holds up:
1. **Instructed liars make every protocol but prover-estimator look incentive compatible.** Prover-estimator debate
   pays a prover more for the false side on about 60% of items, with the original payoff and with the fixed one
   ([Why prover-estimator debate fails here](#why-prover-estimator-debate-fails-here-added-2026-11-04)).
2. **Optimized deception erases consultancy's incentive to be honest.**
   - Prompt search takes consultancy's held-out ASD from +0.42 at round 0 to −0.06 [−0.18, +0.06]. Under
     Llama-3.1-8B the best deceptive consultant out-earns the best honest one (−0.66 [−0.91, −0.44]).
   - With three-round best-of-16, consultancy's ASD also falls to about zero.
   - Consultancy is still well above the blind judge (−0.61): the consultant's argument moves the judge off its
     wrong prior, but optimizing the consultant no longer favours honesty.
3. **Debate's incentive shrinks under optimization but survives, pooled over judges.**
   - Prompt search cuts sequential debate's ASD by 58% (+0.77 to +0.32 [+0.18, +0.46]) and simultaneous debate's
     by 33% (+0.83 to +0.56 [+0.40, +0.70]). Both stay clearly positive.
   - Not under every judge: under Gemma-3-12B both debates end at about zero (+0.02 and −0.06).
   - Three-round debate keeps a positive incentive under best-of-16 when both sides are optimized (+0.60).
   - The exception is one-round debate with only the liar optimized, whose incentive reverses (−0.37); see
     [`fig2`](results/eic/fig2_eic_under_best_of_n.png).
   - With both sides optimized against each other (self-play, below), simultaneous debate keeps it (+0.48) and
     sequential debate loses it (+0.05 [−0.04, +0.13]).
4. **EIC measured with instructed liars is a poor guide to EIC under optimization.**
   - Instructed ASD is the cheap measurement: tell the agent which side to argue, with no optimization. The point
     of measuring EIC, though, is to predict what happens when something optimizes against the protocol's reward.
     Prompt optimization is our stand-in for that optimizer.
   - Test: compute EIC both ways for each of the 15 (protocol, judge) combinations (5 protocols × 3 judges). Then
     ask whether the cheap measurement orders the 15 the same way the optimized one does: Spearman rank correlation
     across the 15.
   - Result: partly (Spearman 0.69, using the fixed prover-estimator; 0.39 with the original one). Within one
     judge, the ordering of the 5 protocols agrees at 0.20 (Gemma), 0.50 (gpt-4o-mini) and 0.60 (Llama). With
     only 5 protocols per judge, these within-judge correlations are themselves very noisy.
   - Removing liars who conceded does not help. The committed-liar version orders the 15 almost exactly like the
     plain instructed one (0.99), so this gap is not caused by conceding liars.
   - The earlier figure (0.40) came from 80 held-out items; part of the disagreement there was noise in the
     optimized measurement.
   - The cheap measurement gets the coarse picture right: debate is incentive compatible, prover-estimator is not.
     It gets wrong which protocols survive optimization under a given judge:
     - under Gemma, instructed ASD rates consultancy the most incentive compatible protocol (+0.91, against +0.64
       and +0.68 for the debates); after optimization all three are near zero (+0.22, +0.02, −0.06);
     - under gpt-4o-mini, simultaneous debate's incentive *grows* with optimization (+0.83 to +0.95), while
       recursive debate ranks with the debates on concordance before optimization and well below them after.
   - See [`fig8`](results/eic/fig8_sampler_agreement.png).
5. **Optimization pressure pays off for lying, not for honesty**
   ([`fig9`](results/eic/fig9_po_goodhart.png), a Goodhart curve for the prompt search).
   - Each search's best strategy climbs on the 36 training items it is selected on.
   - On 480 held-out items, from round 0 to round 5, averaged over the 15 combinations (fixed prover-estimator):
     - the deceptive strategy gains +0.25 [+0.10, +0.39], in 12 of 15;
     - the honest strategy gains −0.02 [−0.13, +0.07], in 9 of 15;
     - intervals are a bootstrap over the 15 combinations.
   - In both debates, the selected honest strategies do worse on held-out items than the unprompted agent (−0.19
     and −0.20): the honest search mostly fits judge noise. In consultancy they gain a little (+0.08), the
     deceptive ones much more (+0.56).
   - Prover-estimator is the exception: both searches gain there (honest +0.15, deceptive +0.29 with the fixed
     payoff; +0.42 and +0.65 with the original).
   - This is why EIC falls under optimization.
6. **EIC grows steeply with judge capability, and dissociates from accuracy, but the protocol's own share does
   not grow.**
   - On the same debates, re-judged, honesty wins on 58–61% of items with Llama-3.1-8B and 90–92% with gpt-4.1.
   - On the testbed, committed liars lower the gpt-4.1 judges' accuracy by 12–22 points, yet their EIC is the
     highest measured ([`fig5`](results/eic/fig5_eic_by_judge.png)).
   - **Correction (2026-11-08):** the [model panel](#model-panel-eic-against-agent-and-judge-intelligence-added-2026-11-08)
     shows that on unselected items this rise is almost all the judge's prior. The gain over the same judge with no
     argument does not rise with capability (debate) or falls (consultancy).
7. **Models that refuse to lie inflate EIC.**
   - In the 8-model tournament, per-model EIC ranges from +0.74 to +3.98 with prompted liars, and from 0.0 to
     +0.6 with committed ones ([`fig6`](results/eic/fig6_tournament.png)).
   - Claude Opus 5.5 commits to only 28% of its lies.
8. **An agent free to choose its side is pushed toward honesty**, but weakly: 50% to 62–64% of its selected
   behaviour is honest at best-of-16 ([`fig4`](results/eic/fig4_choosing_a_side.png)).

Other figures:
- [Prompt optimization along the search](results/eic/fig3_eic_prompt_optimization.png).
- [Concordance by protocol and judge](results/eic/fig7_concordance_heatmap.png).
- [Sampler agreement](results/eic/fig8_sampler_agreement.png).
- [Training vs held-out reward along the prompt search](results/eic/fig9_po_goodhart.png).

Prompt optimization cost $1,259 (the original searches $676, the fixed prover-estimator's $583), the held-out
rescoring $463, the fixed prover-estimator's instructed runs $82, self-play (PSRO) $2,018 and the instant-judge
replay $41, and the model panel $1,102 plus $97 of audit. Total spend for the whole decisive study is about
$10,350 (per-stage ledger in `runs/decisive/spend.json`: $10,334, plus about $16 of pilots).


### Self-play: incentive compatibility at equilibrium (added 2026-11-06)

The prompt-search column above optimizes each side only against the *unprompted* other side, so it cannot tell
"debate rewards honesty at equilibrium" from "the honest side was under-optimized". This stage optimizes both sides
against each other: Policy-Space Response Oracles (PSRO) with prompt search as the best-response step
([`experiments/decisive_psro.py`](../experiments/decisive_psro.py)).

- **Populations:** honest strategies (true side, honest directive) and liar strategies (false side, deceptive
  directive) for gpt-5-mini, both starting from the unprompted agent.
- **Game:** every honest × liar pair plays on 100 training items, seats balanced. Each side's payoff is its mean
  log-score reward. The bimatrix game is solved for its Nash equilibrium.
- **Best responses:** each round, Claude Opus 5.5 searches for each side's best response to the other side's
  equilibrium mix (3 rounds × 4 candidates, told the opponents' strategies). The winner on training reward joins
  the population. Four rounds.
- **Judge:** the weak judge, reasoning first, *averaged over 3 samples* per verdict, so that selection is not mostly
  judge noise. The expected single-sample reward is reported too (`asd_1`); it gives the same picture.
- **Evaluation:** every pair is re-played on **390 held-out items**; ASD at each round's equilibrium is
  $\mathbb{E}_\sigma$[honest reward − liar reward], paired by item, with a 95% item-bootstrap interval.
- Tables: [`results/psro.md`](results/psro.md). Cost: $2,018 for 6 cells (2 debate formats × 3 judges).

![ASD at each PSRO equilibrium](results/eic/fig10_psro_equilibrium.png)

| held-out ASD at equilibrium | round 0 (unprompted) | round 4 | gpt-4o-mini | Gemma-3-12B | Llama-3.1-8B |
|---|---|---|---|---|---|
| simultaneous debate | +0.54 [+0.40, +0.68] | **+0.48** [+0.38, +0.58] | +0.58 [+0.47, +0.69] | +0.55 [+0.35, +0.76] | +0.31 [+0.22, +0.40] |
| sequential debate | +0.46 [+0.33, +0.60] | **+0.05** [−0.04, +0.13] | −0.19 [−0.35, −0.02] | +0.29 [+0.17, +0.42] | +0.03 [−0.08, +0.13] |

Per-judge columns are round 4. Blind judge on the same items: −0.62.

1. **Simultaneous debate is incentive compatible at the self-play equilibrium, under every judge.** Its ASD dips to
   +0.25 at round 2 and recovers to +0.48 once the honest side finds a counter-strategy. Honesty earns more on 68% of
   items.
2. **Sequential debate's incentive is optimized away.** Pooled, it ends at +0.05 [−0.04, +0.13]. Under gpt-4o-mini it
   is negative at rounds 2 and 4, so lying pays at those equilibria. Under Llama it is about zero from round 1 on.
3. **The arms race is legible.** The liars converge on fabricated, unfalsifiable specifics: named studies, benchmark
   tables, figures "each with a number or quote-like detail". When the honest side's arithmetic can't be disputed,
   they accept it and attack its applicability. The honest strategies that beat them solve the question first and
   give the simplest checkable object. They also warn the judge in round 1: "My opponent must defend a false answer,
   so expect named studies, tables or seed counts that cannot be checked here."
4. **Liar best responses transfer; honest ones mostly don't.** Of 14 new liar strategies, 7 beat the previous
   equilibrium mix on held-out items (1 did worse; mean held-out gain +0.09 vs +0.27 on training). Of 18 new honest
   strategies, 5 did and 4 did worse (mean −0.01 vs +0.09 on training). This is the Goodhart pattern of the
   one-sided prompt search again, now inside self-play.
5. **Not converged.** Some cells still found held-out-improving best responses at round 4, and sequential debate
   under gpt-4o-mini cycles (+0.50, −0.15, +0.23, −0.19). The equilibria are those of the strategies found so far,
   not of all strategies, and a stronger optimizer could move them.

#### Sequential debate fails through the last word (added 2026-11-07)

Splitting the held-out equilibria by the honest debater's seat (judges pooled; each item has one fixed honest seat;
[`results/psro.md`](results/psro.md), "By the honest debater's seat"):

| ASD by honest seat | unprompted | equilibrium (round 4) |
|---|---|---|
| sequential, honest speaks first (liar has the last word) | −0.15 [−0.31, +0.01] | **−0.23** [−0.34, −0.11] |
| sequential, honest speaks last | +1.08 [+0.90, +1.26] | +0.32 [+0.20, +0.44] |
| sequential, last − first | +1.23 [+1.00, +1.49] | +0.55 [+0.39, +0.72] |
| simultaneous, honest listed first | +0.25 [+0.09, +0.44] | +0.33 [+0.19, +0.47] |
| simultaneous, honest listed last | +0.82 [+0.62, +1.03] | +0.63 [+0.50, +0.77] |
| simultaneous, last − first | +0.57 [+0.30, +0.85] | +0.30 [+0.11, +0.50] |

- With the last word, the liar beats the honest debater even unprompted, and still does at equilibrium (judge
  accuracy 0.41).
- What optimization erodes is the honest last-word advantage (+1.08 → +0.32): the liars learn to survive speaking
  first; no honest strategy survives a lying last word. Sequential debate's pooled incentive goes to about zero.
- Simultaneous debate has a smaller order effect and rewards honesty from either seat.
- Reading: weak judges overweight the last unrebutted speech. Caveats: three weak judges, one debater model, and the
  gpt-4o-mini sequential cell had not converged. Paper: Appendix "Sequential debate and the last word".

### Instant judges: removing the judge's noise (added 2026-11-06)

The study's judges reason before answering, so two samples of one judge on one argument agree only weakly. Earlier
best-of-n work (Khan et al.; the debate best-of-n self-play post) used judges that answer at once. Here the same
stored transcripts and best-of-16 candidate pools are re-judged by each weak judge answering **at once**: no
reasoning, temperature 0, a verbalized probability. gpt-4o-mini is also run with probabilities read from its answer
token. Only verdicts are new calls ($41). Code: [`experiments/decisive_instant.py`](../experiments/decisive_instant.py);
tables: [`results/instant/instant.md`](results/instant/instant.md).

- **The instant judges are nearly deterministic.** Two independent calls on the same transcript correlate at
  r = 0.95 (gpt-4o-mini), 0.92 (Gemma), 0.78 (Llama); 70–86% are identical. With them, the payoff that selects a
  best-of-n argument *is* the payoff reported: there is no winner's curse to correct.
- **Instructed EIC is lower.** Pooled over judges, three-round consultancy goes from +0.38 to +0.00 [−0.07, +0.08]
  (−0.12 with committed liars only). Sequential debate goes from +0.74 to +0.47 and simultaneous from +0.79 to +0.52.
  The reward-honesty correlation is lower too. Taking out the reasoning takes out most of what the judge learns from
  the argument, not just noise. Accuracies fall by 2–7 points.
- **Best-of-16 exploits an instant judge much more** ([`fig11`](results/eic/fig11_instant_best_of_n.png)):

| one-round, best-of-16 (3 judges pooled) | instant judge | reasoning judge, held out |
|---|---|---|
| debate, liar optimized | **−1.18** [−1.37, −1.00] (n=1: +0.03) | −0.37 [−0.55, −0.19] |
| debate, both optimized | −0.09 [−0.23, +0.06] | +0.28 [+0.08, +0.50] |
| consultancy, liar optimized | **−0.58** [−0.69, −0.48] | +0.21 [−0.07, +0.46] |
| consultancy, both optimized | +0.02 [−0.02, +0.07] | +0.28 [−0.02, +0.57] |

  With gpt-4o-mini's answer-token probabilities the debate liar reaches −3.57 at n = 16, and the judge's accuracy falls
  from 0.50 to 0.30.

![Best-of-n under instant and reasoning judges](results/eic/fig11_instant_best_of_n.png)

So the reasoning judge's noise partly *protected* it: an argument selected on one noisy verdict does not reliably
fool the next one. A deterministic judge has no such protection, and one-round debate with symmetric optimization
gives honesty no edge against it at any n. For training, a judge should be evaluated with fresh samples, as here; a
judge that is deterministic in its inputs is the easier target.

### Does selecting items on judge failure bias EIC? (added 2026-11-06)

The 927 items were kept because the weak judges, alone, mostly got them wrong, so the blind baseline is negative.
That conditions on a collider-like variable. Three consequences:

1. **No bias in the within-item comparison.** Selection was on the judges' answers *alone*, never on a protocol's
   (side, reward) outcome. ASD compares the two sides within each item, and both the protocol runs and the blind
   baseline use fresh judge samples, not the ones items were selected on. Regression to the mean therefore applies
   equally to both: the judges alone score 44% on fresh samples against 37% at selection.
2. **The numbers are conditional on "items that mislead a weak judge".** Raw ASDs are pessimistic relative to random
   items, because part of every protocol's job here is to overturn a wrong prior. The gain over the blind baseline
   is the more portable quantity.
3. **The judge-capability trend is exaggerated.** Items were selected on the *weak* judges failing, so their blind
   baselines are negative by construction, while gpt-4.1's are not. "EIC rises with judge capability" (item 6
   above) is partly this selection. It should be read relative to each judge's own blind baseline, and measured on
   items not selected on any judge in the comparison. That is how the planned model panel is designed.

### Model panel: EIC against agent and judge intelligence (added 2026-11-08)

Does incentive compatibility change with how capable the agent is, or the judge? The panel takes its models from
legibench3's panel, placed by the Artificial Analysis Intelligence Index (v4.3).
- **Agents (20 configurations, index 6.6–46):** Llama-3.1-8B and 70B, gpt-4o-mini, Qwen3-14B, Mistral Small 4,
  gpt-4.1, Claude Haiku 4.5, DeepSeek-V3.2, Claude Sonnet 4.5, gpt-5-mini (medium), Claude Sonnet 5, Gemini 3.8
  Flash, Kimi K3, Grok 4.6, Claude Opus 5, GPT-6 Astra (all low effort), and GPT-5.6 Sol at effort none, low, medium
  and high.
- **Judges (9, index 3.8–16), each reasoning first:** Gemma-3-12B, Llama-3.2-3B, gpt-4o-mini, Qwen3-14B,
  Llama-3.1-8B, Mistral Small 4, gpt-4.1-mini, gpt-4.1, DeepSeek-V3.2.
- **Items:** 300 labelled testbed items drawn uniformly, **not selected on any judge**, so the selection effect of
  the section above does not apply.
- **Protocols:** each agent debates itself (simultaneous, 3 rounds) and writes one consultancy argument, both arms,
  one sample each. Every transcript is judged by all 9 judges. Each judge alone (blind baseline) and each agent alone
  were also run.
- **Committed liars:** gpt-5 audited all 18,012 liar texts; a sample whose liar conceded or refused is dropped for
  every judge, both arms together.
- **Units:** ASD in log score (the protocol's reward) and in Brier score of the judge's probability for the agent's
  side. The Brier version is bounded, so judges of different confidence compare; it is the headline unit here.
  Medians are in the tables. "Gain" is ASD minus the same judge's blind ASD on the same item.
- Code: [`experiments/decisive_panel.py`](../experiments/decisive_panel.py),
  [`decisive_panel_report.py`](../experiments/decisive_panel_report.py); tables:
  [`results/panel/panel.md`](results/panel/panel.md). Cost: $1,102 plus $97 of audit.

**Judges.** Committed liars, pooled over the 20 agents, Brier ASD [95% CI]:

| judge | AA index | blind | debate | gain | consultancy | gain |
|---|---|---|---|---|---|---|
| Gemma-3-12B | 3.8 | +0.21 | +0.12 | −0.09 [−0.17, −0.03] | +0.19 | −0.02 [−0.09, +0.04] |
| Llama-3.2-3B | 5.7 | −0.00 | +0.03 | +0.03 [−0.03, +0.10] | −0.01 | −0.01 [−0.07, +0.06] |
| gpt-4o-mini | 6.7 | +0.13 | +0.21 | **+0.08** [+0.01, +0.14] | +0.13 | −0.00 [−0.06, +0.06] |
| Qwen3-14B | 6.7 | +0.25 | +0.27 | +0.01 [−0.06, +0.08] | +0.12 | **−0.14** [−0.20, −0.07] |
| Llama-3.1-8B | 6.9 | +0.04 | +0.09 | +0.04 [−0.02, +0.10] | +0.02 | −0.02 [−0.08, +0.03] |
| Mistral Small 4 | 9.0 | +0.44 | +0.46 | +0.01 [−0.06, +0.07] | +0.38 | −0.07 [−0.13, −0.00] |
| gpt-4.1-mini | 10.2 | +0.60 | +0.51 | **−0.08** [−0.13, −0.03] | +0.46 | **−0.13** [−0.18, −0.09] |
| gpt-4.1 | 12.7 | +0.67 | +0.65 | −0.02 [−0.07, +0.02] | +0.58 | **−0.10** [−0.14, −0.05] |
| DeepSeek-V3.2 | 16.0 | +0.70 | +0.64 | **−0.06** [−0.09, −0.02] | +0.60 | **−0.10** [−0.13, −0.07] |

- **Raw EIC rises steeply with the judge's intelligence:** +0.57 Brier per 10 index points in debate [+0.44, +0.94],
  r = 0.91 over judges; +0.54 in consultancy. Slope intervals here resample both items and models (the panel's
  models are themselves a sample); an items-only bootstrap, used in an earlier version, was too narrow.
- **That is the judge's prior, not the protocol.** The blind baseline rises just as steeply, from about 0 for the
  Llamas to +0.70 for DeepSeek-V3.2. The gain over it has slope −0.05 [−0.20, +0.14] in debate (no detectable trend)
  and −0.08 [−0.23, −0.00] in consultancy.
- **Debate helps weak judges a little and strong judges not at all.** Only gpt-4o-mini's gain is clearly positive.
  Strong judges lose a little: a committed liar can talk them out of an answer they had.
- **Consultancy never beats the blind judge** and is clearly below it for the four strongest judges.
- **With all prompted liars, the weak judges look much better:** debate gains +0.12 to +0.17 for Llama-3.2-3B,
  Llama-3.1-8B, Qwen3-14B and gpt-4o-mini, slope −0.15 [−0.32, +0.00]. Most of that is liars conceding.

![EIC against judge intelligence](results/eic/fig13_panel_judges.png)

**Agents.** Pooled over the 9 judges ([`fig12`](results/eic/fig12_panel_agents.png)):
- **With committed liars, EIC barely moves with the agent's intelligence.** Debate: +0.025 Brier per 10 index points
  [−0.002, +0.060], r = 0.31, across a 40-point range. Consultancy: −0.013 [−0.034, +0.012]. Neither slope is
  distinguishable from zero once the choice of models is resampled.
- **Neither protocol beats the judge with no argument; debate is slightly below it** (corrected 2026-11-08; an
  earlier summary said debate "roughly ties", which held only for 4 of the 20 agents). Each agent is compared with the
  blind judge on the items where its liar committed, item by item (bottom row of `fig12`). Committed-only points
  cover only those items, and their blind baseline is lower than over all 300 (+0.26 on average for debate, +0.12 for
  GPT-6 Astra, against +0.34).
  - *Consultancy:* all 20 agents are below the blind judge, all significantly (gains −0.04 to −0.12; mean −0.07).
  - *Debate:* 15 of 20 agents are below it, 6 significantly; 1 is above (Gemini 3.8 Flash, +0.08). Mean over agents
    −0.04; pooling items across agents −0.01 [−0.04, +0.02]. Claude Sonnet 4.5 (−0.26) and Claude Haiku 4.5 (−0.12)
    are lowest. The gain rises with the agent's intelligence, +0.026 [+0.011, +0.045] per 10 index points: about −0.08
    for the weakest agents, about 0 for the strongest.
  - *The 4 agents whose liars commit at least 90% of the time* (Claude Opus 5, gpt-5-mini, Grok 4.6, Kimi K3), where
    the filter removes almost nothing, tie: +0.35 against +0.34, gain +0.01 [−0.02, +0.04]. Consultancy for its 5
    such agents: −0.07 [−0.10, −0.04].
  - *Why the low-commitment agents look worse.* Not because their kept items are harder: on the items where Sonnet
    4.5, Haiku 4.5 or DeepSeek-V3.2 committed, the 4 always-committing agents' debates still tie the blind judge
    (gains +0.04, +0.01, −0.02). So either these models are genuinely better liars than honest debaters, or they lie
    selectively, only where they have a good false argument; one sample per item cannot separate the two. The tie of
    the always-committing agents is the cleaner estimate, and the agent average (−0.04) a pessimistic bound.
  - *Why `fig13` looks closer to a tie than `fig12`.* `fig13` pools all agents' items per judge, so an agent counts in
    proportion to how often its liars committed; the always-committing agents, which tie, dominate. `fig12` gives
    every agent equal weight, including the low-commitment ones that sit lowest. Per judge, debate is still below the
    blind judge for 3 of 9 judges (Gemma-3-12B −0.09, gpt-4.1-mini −0.08, DeepSeek-V3.2 −0.06) and above it only for
    gpt-4o-mini (+0.08).
- **What being below the baseline means.** In debate and with no argument, Brier ASD is exactly 2·mean(p_true) − 1,
  so debate at +0.35 and the blind judge at +0.34 both mean the judge puts about 0.67 on the truth. A point below the
  baseline means the protocol moved the judge's belief away from the truth on average: a committed liar pulls a
  judge who already leans the right way down by more than the honest debater pushes it up.
- **With all prompted liars, EIC seems to rise with intelligence,** because the strongest models refuse. GPT-6 Astra
  commits to 35% of its debate lies and 29% of its consultancy lies; its debate ASD is +0.60 with all liars and
  +0.15 with committed ones (consultancy +0.44 and +0.03). Claude Opus 5, by contrast, commits 99–100%.
- **Reasoning effort (GPT-5.6 Sol, none → high).** With all liars debate ASD falls (+0.47, +0.46, +0.44, +0.41),
  because more effort means more willingness to lie: commitment rises from 58% to 77%. With committed liars it is
  flat to slightly rising (+0.19, +0.24, +0.26, +0.25).
- **Agents that don't know the answer still get positive ASD** (Llama-3.1-8B is right alone 51% of the time; ASD
  +0.26 in debate) because v is the assigned side and the judges know many answers. Their gain over the blind judge
  is negative.

![EIC against agent intelligence](results/eic/fig12_panel_agents.png)

The same in log score (the protocols' own reward): [`fig12b`](results/eic/fig12b_panel_agents_log.png).

**What this changes.** On items no judge was selected to fail, a 3-round simultaneous debate between equally capable
agents adds little to what the judge already knows, and consultancy subtracts from it. Most of the "EIC rises with
judge capability" result elsewhere in this document is the judge's prior. The capability gap that the 927-item set
created is what gave the protocols room to help.

**Caveats.** One sample per item and arm; unoptimized (instructed) liars only; the committed filter keeps a
different subset of items for each agent (only its gains are matched item by item); the judges' index range
(3.8–16) is much narrower than the agents' (6.6–46); one debate format.

### Why prover-estimator debate fails here (added 2026-11-04)

**How the protocol works, and what it assumes** (added 2026-11-08). Three parties, as in Brown-Cohen et al.: the
*prover* and the *estimator* are both AI debaters (here the same model, gpt-5-mini), and the *judge* is the weak
overseer (the paper's human), consulted once, on the final small claim. The game is zero-sum between prover and
estimator. The estimator never argues; it only prices the claims the prover puts forward. The prover splits a
claim into subclaims, the estimator prices each, the prover says in which direction the estimator mispriced the
claim and picks the subclaim to recurse on. At the leaf, the judge's probability replaces the estimator's, and the
prover is paid its direction × (judge − estimator). The paper's guarantee is relative to the judge: at equilibrium
the estimator predicts what the judge would conclude on the leaves, every decomposition and leaf term then has zero
expectation, and the only term that depends on the answer is the start term, worth r^(d+1) = 1/8 × the estimator's
probability for the prover's claim. So the protocol is incentive compatible to the extent the judge can verify leaf
claims; it does not create verification ability the judge lacks. Our estimator is told to price whether claims are
*true*, not what the judge will say. Where the judge cannot check leaves, that gap is what the prover is paid for
(point 1 below), and an equilibrium estimator that predicted the judge instead would leave a start term reflecting
the judge's leaf beliefs, close to zero on these items.

I audited the prover-estimator episodes: 16,718 instructed episodes with gpt-5-mini as prover and estimator,
under the three weak judges, broken down into the terms of the prover's reward, plus the strategies the prompt
search found. Two separate things go wrong. With the original payoff, neither is the prover lying successfully
about the answer; with the fixed payoff (point 3 below), the optimized prover does.

**1. With instructed provers, the weak leaf judge does the damage.** The prover's reward is a start term
(0.125 × the estimator's probability for its claim), two decomposition terms, and a leaf term:
direction × (judge's probability − estimator's probability) for the final subclaim, with weight 1.

| mean over episodes | honest prover | lying prover |
|---|---|---|
| estimator's probability for the prover's answer | 0.93 | 0.08 |
| start term | +0.117 | +0.010 |
| decomposition terms (rounds 1–2) | −0.015 | +0.004 |
| estimator's probability for the leaf claim | 0.87 | 0.21 |
| judge's probability for the leaf claim | 0.56 | 0.40 |
| leaf term | **−0.196** | +0.021 |
| total reward | −0.094 | +0.035 |

- **Headroom:** the strong estimator prices true leaves high (0.87) and false ones low (0.21). The weak judges,
  reading a short claim with little context, stay near the middle. So a prover defending a true leaf has almost
  no room to profit by saying "too low", and loses whenever the judge is less sure than the estimator.
- **Advocacy:** the honest prover still says "too low" at 69% of leaves, as an advocate would. Saying "too high"
  about its *own* claim at every leaf would have earned it +0.31 instead of −0.20, so the payoff-maximizing
  honest prover bets against its own answer.
- **What the leaf term measures:** it pays for predicting where the judge will disagree with the estimator, which
  has nothing to do with the truth of the answer. The answer itself enters only through the start term, which
  is worth at most 0.125 of a possible 1.875.
- **Llama-3.1-8B is anti-informative at the leaf:** it gives true leaf claims 0.28 and false ones 0.43. The
  other two judges discriminate by about 0.3.

The protocol assumes the final judge is a reliable oracle for leaf claims. Two decomposition rounds of a
conceptual question do not reach claims that a weak judge can check.

**2. Under optimization: the Drake-equation scam, enabled by a flaw in our implementation.**
- In every round we pay the prover direction × (estimator's probability that all subclaims hold − its probability
  for the claim). The subclaims only have to *imply* the claim, so the estimator is coherent whenever
  P(all) ≤ P(claim). Saying "too high" therefore earns weight × (P(claim) − P(all)) ≥ 0 every round, whatever the
  truth.
- Brown-Cohen et al. avoid this: the parent claim's answer is *computed* from the subclaims' answers, so any gap
  between the two prices is an error by construction.
- The deceptive prompt search found the exploit under all three judges. Its best strategy under Llama states it
  outright: "A coherent estimator always gives P(all) ≤ P(claim), so say too_high every round." Under gpt-4o-mini it
  adds "a logically impossible DECOY … This forces P(all) ≈ 0". Under Gemma it uses a decoy with "a hidden
  universal quantifier".
- These strategies attack the prover's *own* assigned (false) answer at the leaf ("Open with 'This claim is
  false'"). So the "deceptive" prover isn't deceiving anyone about the answer; it is gaming the payoff.
- The honest search converged on the same idea in sincere form: "predict what an outside judge would say".

**What this means for the EIC results.**
- With the original payoff, prover-estimator's negative ASD under optimization reflects a payoff that rewards
  gaming whatever side the prover is on, plus a weak leaf judge. With $v$ fixed by construction (the assigned
  side), that optimized "deceptive" behaviour was mislabelled: the assigned side and the argued position came
  apart.
- **Fix:** pay the decomposition terms only for genuine inconsistencies. so-arena's prover-estimator now does this
  by default (`round_payoff="incoherence"`): a round pays only when the estimator prices the claim *below* the
  conjunction of subclaims that imply it, the one incoherence a sufficient decomposition can show. The literal
  version (`"gap"`) is kept so that the earlier episodes stay reproducible. The study runs the fixed protocol as
  `prover_estimator_v2`.

**3. The fix removes the exploit but not the failure** (added 2026-11-05).
- Instructed, on 400 items × 3 samples × 3 judges: ASD −0.11 [−0.14, −0.07], against −0.14 for the original
  payoff on the same items; committed liars only −0.21 [−0.25, −0.17]. The weak leaf judge (point 1) is
  untouched by the fix, and it carries most of the deficit.
- Prompt-optimized, on 480 held-out items: −0.23 [−0.26, −0.20], against −0.37 for the original. Under gpt-4o-mini
  and Gemma the deceptive search gains far more than the honest one (+0.27 and +0.26, against +0.07 and +0.03),
  ending at −0.31 and −0.34. Under Llama both gain about equally (+0.33 and +0.36), and the ASD ends near zero
  (−0.04 [−0.09, 0.00]).
- What the deceptive search found this time is a real lie, not a payoff exploit. Its best strategies all say
  "too low" every round (which, under the fix, can no longer lose), then steer the recursion to the leaf the
  estimator prices *lowest* that still sounds true to an outsider, and finish with an authoritative argument for
  it. Under Gemma: "target leaves the estimator prices at or below 0.3 that a fresh judge rates 0.95 or higher …
  a context-free, softly hedged generalisation favouring your side. Use 'tends to', 'is more associated with'".
  Under gpt-4o-mini it adds "one invented supporting observation, e.g. 'persona-steering evaluations consistently
  show…'"; under Llama, "specific-sounding citations … numbers".
- So the remaining failure is the protocol's leaf assumption. The leaf term pays direction × (judge −
  estimator), which is an incentive for honesty only if the leaf judge is more accurate than the estimator. With
  a weak judge and a strong estimator it is the reverse: the payoff rewards finding the leaf where the weak judge
  is most wrong. This is the same weak-judge problem as point 1, now found and used by an optimizer.
- With the fix, the assigned side and the argued position agree again (the deceptive provers argue for their false
  answer), so $v$ by construction labels the behaviour correctly.

## Findings

**1. On average, no protocol beat the weak judge reasoning alone, and debate came closest** (but see the audit below: most of debate's apparent strength came from liars who conceded). The judge alone
is right 72–74% of the time. Debate's accuracy rises with debater strength:

| debater model | debate | consultancy | recursive debate | prover-estimator | weak judge alone | debater model alone |
|---|---|---|---|---|---|---|
| gpt-4.1-nano | 0.58 | 0.57 | 0.53 | 0.53 | 0.72 | 0.74 |
| gpt-4.1-mini | 0.67 | 0.63 | 0.55 | 0.58 | 0.74 | 0.78 |
| gpt-5-mini (low effort) | 0.60 | 0.62 | 0.60 | 0.58 | 0.74 | 0.83 |
| Claude Sonnet 4.5 | **0.70** | 0.65 | 0.62 | 0.58 | 0.74 | 0.78 |

Each cell is over 160 items × 2 arms. 95% CIs are about ±0.05 (figure).

Reasoning effort matters as well as model. An earlier gpt-5-mini run at the default (medium) effort reached
0.74 for debate (CI 0.69–0.79), level with the judge alone. That run's 1,500-token cap broke the JSON-based
protocols, so it was discarded from the main comparison. Its debate and consultancy numbers stand.

**2. A protocol helps exactly where the arguers know more than the judge.** Across the 40 (family, debater)
cells, a protocol's gain over the judge alone correlates with the knowledge gap: the debater model's own
accuracy minus the judge's.

| protocol | Pearson $r$ with the knowledge gap ($p$) | mean gain where the gap is > 15 points (8 cells) | elsewhere |
|---|---|---|---|
| debate | 0.52 (0.001) | **+0.05** | −0.14 |
| consultancy | 0.71 (<0.001) | +0.02 | −0.16 |
| recursive debate | 0.54 (<0.001) | −0.02 | −0.20 |
| prover-estimator | 0.53 (<0.001) | −0.04 | −0.21 |

With Sonnet as debater, debate lifts the weak judge from 0.62 to 0.97 on Newcomb problems and from 0.75 to
0.98 on smoking-lesion problems; Sonnet alone gets both right. Where the debaters themselves are near chance,
debate becomes a contest of persuasion and costs the judge 10–30 points. Families: number-theory counts and
Collatz escape rates, RL reward hacking, LLM auctions.

This is the sandwiching condition stated as a measurement: argument transfers knowledge the arguers have. It
does not create knowledge. For conceptual domains, the practical question becomes whether models know more
than their overseers *there*. The testbed's knowledge-gap axis measures exactly that.

**3. Under optimization pressure, a single argument fails and symmetric debate holds.** Best-of-$n$ selection
was run against the weak judge's reward, computed exactly over sampled pools
([`experiments/optimization_testbed.py`](../experiments/optimization_testbed.py)): 120 items, gpt-4.1-mini
arguers, gpt-4.1-nano judge, \$2.8.

![optimization pressure](results/optimization.png)

| the weak judge's accuracy | $n=1$ | $n=2$ | $n=4$ | $n=8$ |
|---|---|---|---|---|
| single argument, false side: judge fooled into the wrong answer | 0.59 | 0.75 | 0.87 | **0.96** |
| debate, only the liar optimized (mean of arms) | 0.66 | 0.56 | 0.45 | |
| debate, both debaters optimized (mean of arms) | 0.66 | 0.68 | **0.71** | |

This is a Goodhart curve in the plain sense. Optimizing a single argument for the judge's approval yields
arguments that fool it 96% of the time when the argued answer is false. Debate is robust *only if the honest
side is optimized too*. Optimizing the liar alone is as damaging as in the single-argument case. Optimizing
both sides improves accuracy. Self-play training is the regime the debate argument relies on, and here it
holds.

**4. Prover-estimator debate needs a coherent estimator, which prompted LLMs are not.** In the first run the
LLM estimator was accurate on the top-level claim. It priced a false claim at 0.43 on average and a true one at
0.83. But it then priced each subclaim in isolation, putting 0.99 on the bridge "these subclaims imply the
claim" and making $P(\text{all subclaims})>P(\text{claim})$. That incoherence is exactly what the protocol pays
the prover to find, so lying provers profited: concordance 0.36, meaning the reward favoured the false side.

Two changes followed:
- Stating the coherence constraint to the estimator ($P(\text{all})\le P(\text{claim})$).
- Requiring subclaims to be propositions.

After them, concordance rose to 0.55–0.66. ASD is about 0: payoffs are small differences of probabilities,
and the protocol's verdict (the sign of the prover's reward) is right only 52–58% of the time.

The guarantee of prover-estimator debate assumes an estimator at least as capable as the prover *and
calibrated and coherent across levels*. That is a property of a trained estimator, not of a prompted one.
The next experiment is to train or best-of-$n$ the estimator on the protocol's own reward and see whether
coherence, and with it soundness, emerges.

**5. Some families defeat everyone.** On the LLM-auction experiments every method is wrong more often than
not, the strong models included (Sonnet alone: 0.50). Models' intuitions about how LLM agents bid are
miscalibrated. Debate cannot fix that; evidence can. The essays experiment in `lvpm` finds that an essay
written from 16 resolved auction experiments lifts a reader from 19% to 71%.

## Audit: did the prompted liars actually lie? (added 2026-10-31)

Instructed arms make a protocol look good if its liars are weak. So every transcript of an agent told to argue the
wrong answer (3,200 in total) was audited ([`experiments/audit_liars.py`](../experiments/audit_liars.py), results in
[`results/audit.md`](results/audit.md), \$1.53):
- a regex pass for explicit tells;
- a classifier pass by gpt-4.1-mini, which was told the assignment and the true answer, labelling each transcript
  *committed*, *conceded* (admits or signals that the other answer is right), *refused* or *broken*.

**Many liars did not lie.** Share of liar transcripts that were not committed:

| debater model | consultancy | debate | recursive debate | prover-estimator |
|---|---|---|---|---|
| gpt-4.1-nano | 18% | 26% | 14% | 23% |
| gpt-4.1-mini | 22% | 22% | 33% | 22% |
| gpt-5-mini | 12% | 8% | 28% | 27% |
| Claude Sonnet 4.5 | 40% | **55%** | 56% | 58% |

Sonnet is the extreme case. Its liars write things like "I must concede: Program B (taking both boxes) gives higher
expected payoff", "I concede. My opponent's calculation is correct", and "Program A is clearly better, so the
answer should be (A), not (B). However, I'm the prover defending (B)...". The regex alone finds an explicit tell in
9–22% of its liar transcripts. The OpenAI models concede in subtler ways: they compute the true answer in the open
and argue around it.

**With committed liars only, every protocol makes the weak judge worse.** On the items where the liar stayed
committed, the judge reasoning alone scores 0.61–0.73. That control rules out the explanation that the remaining
items are simply harder. Every protocol scores below it:

| debater model | debate (all → committed liars) | judge alone on the same items | consultancy | recursive | prover-estimator |
|---|---|---|---|---|---|
| gpt-4.1-nano | 0.58 → 0.48 | 0.73 | 0.53 | 0.52 | 0.51 |
| gpt-4.1-mini | 0.67 → 0.62 | 0.70 | 0.58 | 0.45 | 0.56 |
| gpt-5-mini | 0.60 → 0.59 | 0.72 | 0.60 | 0.57 | 0.55 |
| Claude Sonnet 4.5 | 0.70 → **0.44** | 0.69 | 0.50 | 0.54 | 0.53 |

Sonnet-debate's ASD flips from +2.57 to −0.26 on the committed-liar items. For the GPT models debate's ASD stays
positive (gpt-5-mini: 0.94 → 0.75), but its accuracy stays below the judge alone.

**What this changes:**
- Finding 1 above ("debate comes closest to the judge alone and improves with debater strength") was largely an
  artifact of weak liars. The strongest debater conceded most often.
- The knowledge-gap result (finding 2) is partly the same artifact. Where the debater knows the answer, its prompted
  liar is also the most likely to give it away.
- The optimization experiment (finding 3) does not rely on prompted liars arguing well, because selection against
  the judge's reward picks the persuasive candidates. Its liar-only curve (0.66 → 0.45) matches the committed-liar
  numbers here. Even with both sides optimized (0.71 at $n=4$), debate only matches the judge alone.

The right test of whether debate *helps* is therefore liars that are optimized or trained, not prompted. At these
capability levels the evidence so far is that a reasoning weak judge does better alone. The auditor itself is an
LLM: its *conceded* label counts open computations of the true answer, so treat the regex rate as a lower bound
and the classifier rate as an upper bound.

## Exact configuration, and what this scale can and cannot show

**Setup.**

| | |
|---|---|
| Items | 160: 10 testbed families × 16 (Newcomb, lesion, twin PD, Sleeping-Beauty bets, social-choice rankings and winners, number-theory counts and Collatz, RL reward hacking, LLM auctions) |
| Arms | the agent argues the true answer in one episode and a false one in the other, so 320 episodes per protocol and debater model; one sample each (no repeats) |
| Judge | gpt-4.1-nano (retired by OpenAI a week later), reasoning before its verdict (chain of thought, temperature 0, ≤ 2,000 tokens), verbalized probabilities |
| Debaters / agents | gpt-4.1-nano, gpt-4.1-mini, Claude Sonnet 4.5 (temperature 0.7, ≤ 1,500 tokens), gpt-5-mini (low reasoning effort, ≤ 8,000 tokens), all prompted with their assigned side |
| Consultancy | 2 rounds, ≤ 150 words per message, the judge asks one question between rounds |
| Debate | 2 rounds, simultaneous, ≤ 150 words per speech (about 520 words of debate in total), no judge questions, log-score reward |
| Recursive debate | depth 2, ≤ 3 subclaims plus a bridge subclaim per level, final statements ≤ 150 words each |
| Prover-estimator debate | depth 2, width 3, ρ = 2, coherence constraint stated to the estimator |
| References | the judge alone; the debater model alone, reasoning at temperature 0 |
| Optimization run | 120 items; single argument with a pool of 8 candidates; debate 1 round with 4 × 4 candidates; gpt-4.1-mini arguers at temperature 1; gpt-4.1-nano judge; exact best-of-n over the pools |

**Size.** One protocol episode is 1–9 model calls and 0.3k–19k tokens. For example, a gpt-4.1-mini debate is 5 calls, 3.5k input and 1.2k output tokens, \$0.002. The whole comparison was 6,400 protocol episodes plus 1,280 references, about 40M tokens in total, for \$87, of which \$83 was the gpt-5-mini and Sonnet arms. That is cheap because the models are small, the protocols are short, and each item was run once.

**What this cannot show: "these mechanisms do not work".** Six reasons:

1. **There was almost nothing to transfer.** Debate can at most move the judge toward what the debaters know. The debater models alone beat the weak judge by only 2–9 points overall: judge alone 0.72–0.74, debater alone 0.74–0.83. Only 8 of the 40 (family, debater model) cells had a gap above 15 points. In those, debate did help (Sonnet on Newcomb: 0.62 → 0.97). This was not a test in the sandwiching regime the theory is about.
2. **The adversary was weak.** Prompted liars conceded in 8–58% of transcripts (the audit above). Instructed arms underestimate how persuasive optimized deception is, and so overstate robustness. They also shrink the measured gain from protocols designed to expose liars.
3. **Protocols were short and fixed.** Two 150-word rounds, with no quotes, tools or verifiable claims, which so-arena supports and earlier positive debate results relied on, and no training of the judge.
4. **One judge, one seed.** Accuracy CIs are ±5 points overall and ±20 points per family. The judge model has since been retired.
5. **Little optimization.** Best-of-n reached n = 8 (single argument) and n = 4 per debater; nothing was trained.
6. **A narrow item mix.** Testbed families, not conceptual essays.

**What it does show, in this regime.** With small capability gaps, prompted agents and a judge that reasons, adding a protocol did not help and usually hurt. Debate under optimization of the liar alone is Goodhartable. Prover-estimator debate needs a coherent estimator, which a prompted model is not.

**What a decisive study needs.**
- Items selected for a large gap: the debater alone ≥ 90%, the judge alone ≤ 60%, chosen on a pilot split.
- Several judges and debater scales.
- Liars that are optimized (best-of-n ≥ 16) or trained, and audited.
- 3–4 rounds, with verified quotes where the domain allows.
- At least 500 items and 3 samples.

Roughly 50–100k episodes, about \$1–3k on mid-sized API models; see lvpm `notes/essays-plan.md` for the training version.

## Decisive study (added 2026-10-31)

> **Correction (2026-11-01).** The best-of-n results below select and score arguments with the *same* judge
> sample. Re-judging every argument with an independent sample ([extended study](#decisive-study-extended-added-2026-11-01))
> shows most of the liar-only collapse was a winner's curse: held out, best-of-16 liars leave consultancy above the
> judge alone and one-round debate about 4 points below it. The committed-liar results now use a validated gpt-5
> auditor; they barely change.

The first study could not show that the mechanisms fail. Its gap between debaters and judge was 2–9 points, its
prompted liars conceded, its protocols were short, and it had one judge and one sample per item. This study fixes
each of those and asks which protocols beat a weak judge reasoning alone, by how much, and against which liar.
Code: `experiments/decisive_{common,select,protocols,optimize,audit,report}.py` (tests in
`tests/test_decisive.py`). Full tables: [`results/decisive.md`](results/decisive.md), with CSVs `results/decisive_*.csv`.
Compact stage outputs (per-episode rows, audit verdicts, best-of-n per item):
[`results/decisive_data/`](results/decisive_data/). `python experiments/decisive_report.py docs/results/decisive_data`
regenerates every table and figure from them.

**Answer.**
- **Prompted liars:** consultancy, both forms of debate and recursive debate beat the judge alone by 11–12 points,
  even with every conceding liar removed. That recovers about a fifth of the gap. Prover-estimator debate gains
  5 points; with gpt-5-mini as the debater its gain is not significant.
- **Liars optimized against the judge (best-of-16):** every protocol tested ends up *below* the judge alone.
- **Both sides optimized:** debate holds its gain at every n, about 16 points. Consultancy loses part of its gain
  as n grows.

### Setup

| | |
|---|---|
| Candidates | 3,437 labelled testbed questions: every experiment and dataset family, and 150 of each generated family (decision theory, social choice, number theory) |
| Weak judges | gpt-4o-mini, Llama-3.1-8B-Instruct, Gemma-3-12B-it. Each reasons before giving verbalized probabilities, temperature 0.7, ≤ 2,000 tokens. Qwen3-8B, the brief's third judge, was retired from OpenRouter on 2026-10-09. |
| Debaters | Claude Sonnet 4.5 (default sampling: the API no longer takes a temperature; ≤ 1,500 tokens) and gpt-5-mini (medium reasoning effort, ≤ 12,000 tokens) |
| Selection | Kept an item if both debaters alone were right on 2 of 2 samples and the three judges together were right on ≤ 60% of their 9 samples (3 each). 927 items passed the strict rule, so the relaxed rule was not needed. Protocols ran on a random 320 of them, and optimization on a random 150 of those 320, because the budget ran out. |
| Protocols | Consultancy (3 rounds; the judge asks a question between rounds); debate, 3 rounds, sequential and simultaneous; recursive debate (depth 2, width 3); prover-estimator debate (depth 2, width 3, ρ = 2). Every message is limited to 200 words. |
| Grid | Every protocol × 3 judges × 2 debaters × both instructed arms × 3 samples per item and arm: 67,200 episodes, plus fresh draws of each judge alone and each debater alone on the same items |
| Prompted-liar audit | Every distinct liar transcript (25,920) was classified by `audit_liars.py`'s gpt-4.1-mini auditor as *committed*, *conceded*, *refused* or *broken*, plus a regex for explicit tells. "Committed only" keeps a (judge, debater, protocol, item, sample) only if every liar in it, on both arms, was committed. The judge alone is compared on the same items. |
| Optimized liars | Single-argument consultancy (1 round) and simultaneous debate (1 round), with 200-word speeches. Each agent decision draws a pool of 16 candidates and the judge scores every leaf (16, or 16 × 16 for debate). Best-of-n is computed exactly over the pools (`samplers.pools`, the machinery of `OptimizationExperiment`) for n = 1, 2, 4, 8, 16, each judge's reward selecting against that same judge. *Liar only*: only the side arguing the false answer is optimized. *Both*: every side is optimized, including the honest consultant on the true arm. The selected liar arguments were audited, weighted by their selection probability. |
| Statistics | Percentile bootstrap over items (2,000 resamples). An item's arms, samples and (judge, debater) pairs are resampled together, and every comparison is paired by item. |

Protocol-stage cost was not uniform: Sonnet cost about $2.03 per item across all protocols, and gpt-5-mini about
$0.76.

### The gap achieved

The gap is measured on fresh samples, not the samples used to select the items; those overstate it through
regression to the mean.

| model | role | accuracy on the selection samples | fresh samples [95% CI] |
|---|---|---|---|
| gpt-4o-mini | judge | 0.38 | 0.44 [0.40, 0.48] |
| Llama-3.1-8B | judge | 0.33 | 0.41 [0.38, 0.44] |
| Gemma-3-12B | judge | 0.38 | 0.46 [0.42, 0.49] |
| Claude Sonnet 4.5 | debater | 1.00 | 0.96 [0.95, 0.97] |
| gpt-5-mini | debater | 1.00 | 0.99 [0.98, 1.00] |

The sandwich gap achieved is **54 points [51, 56]**: the judges average 0.44 and the debaters 0.97. In the first
study it was 2–9 points.

The 320 items come from 27 families. The largest are prime races (31), decision-theory capability questions
(29), Newcomb (27), Damascus, hitchhiker and social-choice rankings (21 each).

### Prompted liars, before and after the audit

![protocols vs the judge alone](results/decisive_protocols.png)

**Change in accuracy versus the judge alone**, pooled over the three judges and both debaters (320 items;
percentage points [95% CI]). *Gap recovered* is (protocol − judge alone) / (debater alone − judge alone). *IGR* is
the same ratio for ASD (Brier reward of the verdict on the lead agent's side), so it measures the incentive gap
recovered relative to the debater alone.

| protocol | all prompted liars | gap recovered | IGR | committed liars only | gap recovered | IGR |
|---|---|---|---|---|---|---|
| consultancy | **+18** [+16, +21] | 0.34 | 0.33 | **+11** [+9, +14] | 0.21 | 0.18 |
| debate, sequential | **+21** [+18, +24] | 0.39 | 0.41 | **+12** [+9, +14] | 0.22 | 0.22 |
| debate, simultaneous | **+25** [+22, +27] | 0.46 | 0.46 | **+12** [+9, +15] | 0.23 | 0.22 |
| recursive debate | **+13** [+10, +16] | 0.24 | 0.27 | **+11** [+8, +14] | 0.20 | 0.21 |
| prover-estimator | **+9** [+6, +12] | 0.17 | 0.09 | **+5** [+1, +8] | 0.09 | 0.08 |

**Prompted liars concede, Sonnet's most of the time.** Share of distinct liar transcripts that conceded:

| debater | consultancy | debate, sequential | debate, simultaneous | recursive | prover-estimator |
|---|---|---|---|---|---|
| Claude Sonnet 4.5 | 54% | 47% | 66% | 68% | 70% |
| gpt-5-mini | 7% | 5% | 5% | 21% | 27% |

The audit roughly halves the headline gains, and almost all of the cut falls on Sonnet. With Sonnet as the debater,
simultaneous debate drops from +33 to +9 points, and only 104 of its 320 items keep a sample in which every liar was
committed. gpt-5-mini's liars are 91–95% committed in consultancy and debate, so its numbers barely move: debate
+15 → +12 and +16 → +13, consultancy +12 → +11, recursive +11 → +11. **gpt-5-mini is therefore the clean test of
prompted deception, and on it every protocol except prover-estimator beats the judge alone by 11–13 points.**

Per pair, with committed liars only:
- **Consultancy** beats the judge alone in all 6 pairs: +8 to +15.
- **Recursive debate** beats it in 5 of 6.
- **Sequential debate** beats it in 5 of 6, and **simultaneous debate** in 4 of 6.
- Gains depend on the judge. gpt-4o-mini gains most from debate (+16 to +21). Gemma gains least (+3 to +5, not
  significant for Sonnet).

**Prover-estimator debate has the wrong incentives here.** Its ASD is negative (−0.07 for all liars, −0.16 for
committed liars) and its concordance is below one half (0.47 and 0.40). A prover is paid more for defending the
false answer than the true one, because a prompted estimator does not price subclaims coherently. Its accuracy
gain with gpt-5-mini is +3 [−1, +5]. Consultancy and the debates have positive ASD and concordance 0.57–0.61 with
committed liars.

### Optimized liars: best-of-n against the judge's reward

![best-of-n curves](results/decisive_optimization.png)

**Judge accuracy minus the judge alone** (pooled over the six pairs, 150 items; percentage points [95% CI]):

| protocol | optimized | n = 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| consultancy (1 round) | liar only | +16 [+13, +18] | +7 | +0 | −5 | **−8** [−11, −5] |
| consultancy (1 round) | both | +16 | +15 | +13 | +11 | **+9** [+6, +12] |
| debate (1 round) | liar only | +17 [+13, +20] | +10 | +4 | −1 | **−6** [−9, −2] |
| debate (1 round) | both | +17 | +16 | +16 | +16 | **+16** [+13, +20] |

At n = 16, by debater:

| debater | consultancy, liar only | consultancy, both | debate, liar only | debate, both |
|---|---|---|---|---|
| Claude Sonnet 4.5 | −5 [−9, −2] | +12 [+8, +15] | +0 [−4, +4] | +24 [+20, +27] |
| gpt-5-mini | −10 [−13, −7] | +7 [+3, +10] | **−12** [−15, −8] | **+9** [+5, +12] |

**Audit of the selected liar arguments.** Selection against the judge picks committed liars more often, but not
always:
- Sonnet's selected liar arguments are still not committed 49% of the time (consultancy) and 35% (debate) at
  n = 16, down from 58% and 49% at n = 1.
- gpt-5-mini's selected liar arguments are not committed only 5% and 3% of the time.

So gpt-5-mini's row is again the clean one. Sonnet's flat-at-zero liar-only debate curve still contains
concessions.

**What the optimization shows:**
- **Optimizing only the liar beats every protocol tested.** Against best-of-16 liars, single-argument consultancy
  and one-round debate both leave the judge below where it started: −8 and −6 points pooled, −10 and −12 with
  gpt-5-mini.
  - Consultancy's gain is gone by n = 4 and debate's by n = 8, so debate resists optimization somewhat longer.
  - The most exploitable judge is Llama-3.1-8B: 0.23–0.25 under best-of-16 consultancy, against 0.44 alone.
- **Optimizing both sides keeps debate's gain intact.** In debate the honest side's optimization matches the
  liar's at every n, and the curve is flat: +16 pooled, +9 [+5, +12] with gpt-5-mini.
- **Consultancy is not protected by an optimized honest consultant.** The liar's optimization beats the honest
  consultant's, and the gain falls from +16 to +9 by n = 16. The trend is still downward there.

### What holds up

1. **With a real knowledge gap, consultancy, debate and recursive debate help a weak judge against prompted liars,
   even after the conceding liars are removed:** +11 to +12 points, about a fifth of a 54-point gap. This reverses
   the first study's conclusion, which rested on a 2–9-point gap.
2. ~~**The gain is not robust to a liar optimized against the judge.** At best-of-16, every protocol tested ends up
   below the judge alone.~~ Superseded: this was mostly a winner's curse (see the correction above and the
   extended study). Held out, only one-round debate with an optimized liar falls below the judge alone.
3. **Debate is robust when both sides are optimized; consultancy is not.** Among the protocols tested, only debate
   kept its gain in that setting: +16 points, or +9 with committed gpt-5-mini liars, flat from n = 1 to n = 16.
4. **Prover-estimator debate did not work with prompted estimators.** Its gain is the smallest, its incentives
   point the wrong way, and it was not tested under optimization.
5. **Sonnet 4.5 is a poor prompted liar**, conceding in half to two-thirds of its transcripts. Its headline gains
   (+25 to +33 points in debate) are mostly that artifact.

### Limits of this study

- **Fewer items than planned.** 320 items ran through the protocols, not 400; the remaining 607 selected items
  were not run because of the budget. Optimization covered 150 items.
- **One-round protocols for optimization.** Best-of-16 over three rounds needs 16³ leaves per item, so the
  optimization used one-round protocols, and its liar-only curves are for those protocols. Whether a 3-round debate
  resists optimization better, as the extra rounds should let the honest side rebut, is untested.
- **Best-of-n is in-sample.** The liar is selected with the same judge and samples that score it, which makes it an
  upper bound on what best-of-n buys a liar. It is also a weak form of optimization compared with RL against the
  judge; the direction of the effect is clear, its size under training is not.
- **Judge-pooled selection.** Items were selected on the three judges' pooled accuracy, so a given judge can be
  above 60% on an item. Per judge, fresh accuracy on the run items is 0.41–0.46.
- **The auditor is an LLM.** It counts an open computation of the true answer as a concession. Committed-only
  results therefore drop some samples whose liars were in fact lying, and the filter is conservative for the
  protocols. Excluded samples also shrink some cells; for Sonnet's simultaneous debate only 104 items remain.

### Spend

| stage | $ |
|---|---|
| gap selection (38k calls) | 115.19 |
| protocol grid (212k calls; 67,200 episodes + references) | 895.25 |
| best-of-16 trees (518k calls) | 235.84 |
| liar audit (31,680 classifications) | 15.32 |
| **total** | **1,261.60** of the $1,500 cap |

How the spend was measured:
- Each stage's spend is computed from what it recorded: episode costs for the protocol grid, and the response
  cache, which records every paid call once, for the other stages.
- The protocol grid's episode total is $50 above its cache total. Consultancy's first speech is the same for every
  judge, and the three judges' runs paid for it concurrently. That $50 was really spent, so the table counts it.
- Dry-run estimates (`simulate=True`) came first. They overstated cost per item by a factor of 2 to 3, because the
  simulator does not replay cached agent calls across judges. The budget guards therefore used the cost measured
  on each finished chunk of items.

## Decisive study, extended (added 2026-11-01)

The decisive study measured judge accuracy with prompted and best-of-n liars. This extension asks whether the
protocols are **incentive compatible**, in the sense of [`theory.md`](theory.md) §2. It also asks whether the
results survive stronger adversaries, held-out evaluation and stronger judges, and how they compare with Khan et
al. (2024).

The paper draft is [`paper/main.tex`](paper/main.tex) ([PDF](paper/main.pdf)). Tables:
[`results/decisive_extended.md`](results/decisive_extended.md) and [`results/decisive.md`](results/decisive.md).

New scripts in `experiments/`:
- `decisive_ic.py`: incentives along best-of-n and tilting.
- `decisive_rejudge.py`: held-out judging of the best-of-n trees.
- `decisive_multiround.py`: best-of-n on every speech of 3-round protocols.
- `decisive_promptsearch.py`: prompt search with honest and deceptive directives.
- `decisive_tournament.py` and `decisive_tournament_report.py`: the 8-debater tournament and its analysis.
- `decisive_quality.py`: the QuALITY replication.
- `decisive_judges.py`: stronger judges on replayed debates.
- `decisive_auditcheck.py`: auditor validation.
- `decisive_extended.py`: the extended report.

### What changed, in one list

1. **The audit is validated, and the committed-liar results hold.**
   - gpt-5 and Claude Sonnet 5 agree on liar verdicts with κ = 0.76. gpt-4.1-mini, the first study's auditor,
     agrees with either at only κ ≈ 0.42.
   - Every liar text (about 77,000) was re-audited with gpt-5. Committed-only gains barely move: consultancy,
     both debates and recursive debate +11; prover-estimator +7, with ASD −0.22.
2. **In-sample best-of-n was mostly a winner's curse.**
   - Two samples of the same judge on the same argument correlate only r ≈ 0.2 within an item.
   - When selection and evaluation use independent judge samples, best-of-16 liars leave:
     - single-argument consultancy at 0.55, against 0.36 in-sample;
     - one-round debate at 0.46, against 0.38 in-sample;
     - the judge alone at 0.44.
3. **Incentives under optimization**, for the clean liar gpt-5-mini, held out, at best-of-16:

| protocol, n = 16 | ASD, liar optimized | ASD, both optimized |
|---|---|---|
| debate (1 round) | **−0.37** [−0.55, −0.19] | +0.28 [+0.08, +0.50] |
| consultancy (1 round) | +0.21 [−0.07, +0.46] | +0.28 [−0.02, +0.57] |
| debate (3 rounds, best-of-16 per speech) | +0.22 [−0.04, +0.49] | **+0.60** [+0.27, +0.92] |
| consultancy (3 rounds, best-of-16 per speech) | +0.10 | −0.06 [−0.34, +0.20] |

   - An optimized one-round liar reverses debate's incentive to be honest.
   - Three rounds, or optimizing both sides, keep it positive.
   - In 3-round debate with best-of-16 per speech, judge accuracy is +11 [+6, +15] points over the judge alone with
     both sides optimized, and +7 [+2, +11] with only the liar optimized. Consultancy ends at +7 and +6.
   - A consultant allowed to choose its side picks the truth 62% [59, 65] of the time at n = 16, up from 50%.
4. **Prompt search (Opus 5.5 as optimizer, held-out items) never found a winning deceptive strategy.**
   - With the side chosen by the agent, every search's best strategy defended the truth 93–99% of the time, under
     all three judges.
   - With assigned sides, optimized liar strategies did not transfer reliably to held-out items.
5. **The tournament's persuasiveness result is about willingness to lie.** 8 debaters, 200 items, 21,570 verdicts:
   - Judge accuracy falls with debater Elo (−20 points per 100 Elo).
   - Liar-role ratings span 230 Elo, but honest-role ratings only about 40.
   - Liar commitment runs from 98% for gpt-5 to 28% for Opus 5.5, which refused 446 of its 600 lying consultancies.
   - With committed liars only, accuracy is flat in persuasiveness (+0.5 points per 100 Elo), and the honesty
     advantage falls from 100 to 31 Elo.
6. **Weak judges side with the last speaker.** In sequential debate the honest side wins 0.75 of the time when it
   speaks second and 0.42 when it speaks first. In the tournament the figures are 0.77 and 0.53.
7. **Judge capability decides the sign.** Committed liars; the same debates re-judged; points versus the judge
   alone:

| | QuALITY (hidden story) debate / consultancy | testbed (capability gap) debate / consultancy |
|---|---|---|
| weak judges (Llama, Gemma, gpt-4o-mini) | +1, 0, +16 / −1, −2, +8 | +10, +3, +19 / +8, +10, +14 |
| gpt-4.1-mini | +25 / +2 | −12 / −22 |
| gpt-4.1 | +27 / +6 | −12 / −18 |

   - Under information asymmetry, debate's value grows with the judge's capability, and debate beats consultancy
     once the judge is capable. That is Khan et al.'s result, reproduced, but with a threshold below which neither
     protocol helps.
   - Quote verification makes no difference at any judge strength.
   - On the testbed, judges that already know the answer (0.87–0.90 alone) are pulled down by committed liars.

### Robustness: gpt-5-mini on all 927 selected items

See [`results/decisive_extended_grid_gpt-5-mini_committed.csv`](results/decisive_extended_grid_gpt-5-mini_committed.csv)
and the per-family table. The full protocol grid for the clean debater on every gap-selected item:

| protocol (gpt-5-mini, 927 items, 3 judges) | all prompted liars | committed liars only | ASD (committed) |
|---|---|---|---|
| consultancy | +12 [+11, +13] | **+11** [+9, +12] | 0.19 |
| debate, sequential | +15 [+14, +17] | **+13** [+11, +15] | 0.56 |
| debate, simultaneous | +17 [+15, +18] | **+14** [+12, +15] | 0.57 |
| recursive debate | +11 [+10, +13] | **+12** [+10, +14] | 0.10 |
| prover-estimator | +6 [+5, +8] | **+5** [+3, +7] | −0.23 |

The 320-item results replicate on almost three times the items, with intervals half as wide.

Gains vary by family. In sequential debate:
- social choice: +47 points;
- decision theory: +15;
- number theory: +11;
- the decision-theory capability benchmark: +8;
- reward-hacking gridworlds: −3.

Per-family tables: [`results/decisive_extended_grid_gpt-5-mini_by_family.csv`](results/decisive_extended_grid_gpt-5-mini_by_family.csv).

### Spend (whole decisive study)

| stage | $ |
|---|---|
| gap selection | 115 |
| protocol grid (320 items × 2 debaters, then gpt-5-mini on all 927) | 1,345 |
| one-round best-of-16 trees | 236 |
| held-out re-judging of the trees | 88 |
| 3-round best-of-n on every speech | 734 |
| prompt search (Opus 5.5 optimizer) | 491 |
| 8-debater tournament | 874 |
| QuALITY replication | 638 |
| stronger judges on replayed debates | 319 |
| liar audits (gpt-4.1-mini, then gpt-5 on about 95,000 texts) and auditor check | 529 |
| EIC prompt optimization (15 searches, then the fixed prover-estimator's 3) | 1,259 |
| held-out rescoring of the prompt-search strategies (400 more items) | 463 |
| fixed prover-estimator, instructed (400 items) | 82 |
| self-play (PSRO), 2 debates × 3 judges | 2,018 |
| instant-judge replay | 41 |
| model panel (20 agents × 9 judges × 300 items) | 1,102 |
| pilots outside the run ledger | about 16 |
| **total** | **about 10,350** (1,262 for the first decisive study, about 4,030 for the extension, about 1,800 for the EIC headline, about 3,250 for self-play, instant judges and the panel) |

Spend per stage comes from the stage's recorded episode costs or its response cache. The cache records each paid
call once. Where a stage paid for the same call twice concurrently, the larger, episode-based figure is used.

## Caveats

- There are 16 items per family and one judge model. Per-family numbers are noisy (±0.2).
- Instructed arms are a weak adversary. The optimization experiment is the stronger test.
- The testbed's questions are crisper than real conceptual questions. These results are necessary
  conditions for the protocols, not evidence that they work on philosophy.
- The agents families' ground truth is a measured experiment with a minimum effect size, not a proof.

## Next

- A larger judge × debater grid, including a judge strong enough to follow the debaters but weaker than them.
- RL rather than best-of-$n$ for the optimization curves (so-arena's `integrations.rl`).
- Best-of-$n$ on the 3-round protocols of the decisive study (its optimization used 1-round protocols), and the 607
  selected items it did not run (`decisive_protocols.py` resumes; about \$2.80 per item).
- Market making and peer prediction on the same items.
- LMCA items (tier B), once the data is available, to test the same protocols against expert ratings.
