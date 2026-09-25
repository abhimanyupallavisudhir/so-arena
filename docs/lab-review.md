# Review of the supplied lab proposal

Reviewed document: **Oversight That Scales**, dated 17 September 2026, 84 pages; the user supplied it as `solab-application-ideas.pdf`. The research design sections, especially pp. 29–46, directly inform this library. The report is a secondary synthesis, not an empirical result or a binding specification. This review does not independently verify its funding, pricing, personnel, benchmark or legal claims.

## Decisions carried into the implementation

| PDF section | Implication | Implemented support |
| --- | --- | --- |
| §6.3, paired branches | Hold starting state fixed and isolate every mutable dependency | Fresh environment context per arm, snapshot identity, paired seeds, rejection of unspecified stateful execution |
| §6.4, graded ASD | Binary honesty labels do not cover real work | Multidimensional independent measurements; within-task covariance, concordance and selection regret |
| §§6.2–6.5, audit the reviewer | An LLM reviewer is the mechanism under test, not automatically truth | Separate reward/evaluation types and exact/proxy/human/outcome provenance |
| §§6.3–6.4, manipulation failures | Assigned deception may fail or become honesty under optimization | Intent-only versus independently audited ASD; exclusions and optional constrained search |
| §7.1, scaling curves | Distinguish capability, compute, model family and judge access | Model covariates, arbitrary parameter grids, optimization curves and CSV |
| §§7.2–7.3, training | Prompted deception and actual learning have different support | Natural/stratified sampling, prompt search, nested BoN, tabular RL and external trainer export |
| §8.1, explanation | Reward understanding and rejection of errors, not mere agreement | Arbitrary role rewards and evaluation dimensions; blinded human packets and timed rating provenance |
| §8.2, library | Reuse evaluation infrastructure | Actual Inspect policy/task/log integration and ControlArena score mappings |
| §8.3 and Table 8 | Use real expertise gaps and scoped verification | Private SQL suites, chess legality/engine components, delayed forecast scoring, generic artifact evaluators |

The baseline protocols remain domain-independent. Domain-specific verifiers and question formats are supplied explicitly. That is the broad extension mechanism for future experiments, rather than adding a separate experiment engine each time a new domain appears.

## Corrections and caveats that affect library design

The PDF flags a possible historical Propaganda prompt-assignment bug and an ambiguity between Brier conventions (§1 and appendix D). Those historical claims were not independently established here and should be checked against the cited original code and run artifacts before being repeated as facts. This implementation addresses the general failure modes: record effective private instructions, test intervention delivery, publish score definitions, and independently audit realized behavior. Its summed binary negative Brier convention is $-2(1-p)^2$ for endorsement and $-2(p-y)^2$ for event outcomes.

The report sometimes calls truth “known by construction” while also explaining the need for manipulation checks. The latter qualification is essential. An instruction to lie is not a truth label, and optimizing a lie prompt may produce an honest answer. Reference-answer equality also cannot establish whether a false response was intentionally deceptive.

Its centered graded formula is an unnormalized covariance-like statistic over a chosen behavior set. Changes to the support or its weights change the statistic. The library exposes covariance and pairwise measures without presenting either as a universally comparable generalized ASD. It also records the distinction between mechanism reward and task success.

The PDF's chess discussion correctly motivates a narrow expert/generalist comparison but notes that different Elo pools are not directly comparable. `ModelSpec.domain_capability` stores calibrated measurements rather than inventing an Elo gap from parameter counts or labels. Engine-node budgets and version are explicit; engine evaluations are labelled proxies. The table's costs, puzzle counts and reported model solve rates are not hard-coded into the library.

The PDF recommends an oversight layer on Inspect. The implementation follows that direction with a standalone core and optional adapters, avoiding a mandatory dependency on rapidly changing ControlArena internals. Existing ControlArena scorers need explicit mapping; post-hoc log analysis alone does not establish robustness to training.

## What remains experimental work

The library can express the six strands, but does not claim to have run them. A realistic firm environment still needs pinned repositories, database snapshots, simulated customers, mail/payment sinks, audit criteria and dataset governance. Human explanation studies need a recruited panel, study procedures and held-out transfer tasks. Neural RL needs a selected trainer and compute. Real model comparisons require prespecified models, budgets, tasks and independent evaluators. Forecast studies require resolution rules before collecting predictions.

For a first study, use executable SQL or chess items to validate reward/quality separation and paired manipulation checks, then measure optimization curves with actual models and held-out judge draws. In parallel, import a small ControlArena log corpus with known provenance and audit whether the available evidence supports an individual reward interpretation. Compare matched-cost mechanisms before expanding task horizons. This is a suggested research sequence, not a completed empirical claim.

The funding proposal, incorporation advice and imminent grant deadlines in the document are outside this software task. No external application, message, account creation or spend was initiated. The source PDF remains in the task's durable attachments; only its technical implications and hash are included here.
