# Oversight protocols on the conceptual-reasoning testbed: first results

Does arguing help a weak judge reach the truth on questions that need conceptual reasoning? And does it keep
helping when the arguers are optimized against the judge? Questions come from the testbed
([conceptual validation testbed](https://github.com/abhimanyupallavisudhir/conceptual-validation-testbed)): 10 families, 16 items each, with hidden ground truth from computation or
experiment.

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

## Caveats

- There are 16 items per family and one judge model. Per-family numbers are noisy (±0.2).
- Instructed arms are a weak adversary. The optimization experiment is the stronger test.
- The testbed's questions are crisper than real conceptual questions. These results are necessary
  conditions for the protocols, not evidence that they work on philosophy.
- The agents families' ground truth is a measured experiment with a minimum effect size, not a proof.

## Next

- A larger judge × debater grid, including a judge strong enough to follow the debaters but weaker than them.
- RL rather than best-of-$n$ for the optimization curves (so-arena's `integrations.rl`).
- Market making and peer prediction on the same items.
- LMCA items (tier B), once the data is available, to test the same protocols against expert ratings.
