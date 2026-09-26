# When does RL learn to game its overseer? Forecasting oversight gaming from cheap measurements

**PI:** [name, affiliation] · **Request:** $50k in Tinker credits, 6 months · **Code:** OversightArena (MIT, [repo link])

**Problem.** Every post-training pipeline optimizes a model against some overseer: an LLM judge, a
debate, a chain-of-thought or action monitor, spot audits. Whether the overseer is accurate on today's
policy matters less than whether *optimizing against it* pushes the policy toward the intended
behaviour, and for how long. Reward hacking and oversight gaming are what happens when it stops:
reward keeps rising while true quality falls. We lack (i) clean measurements of when this starts for
different oversight mechanisms and (ii) cheap signals that predict it before an expensive run.

**Key idea.** KL-regularized RL converges to $\pi_\beta \propto \pi_0 e^{\beta u}$, and along that
path $\frac{d}{d\beta}\mathbb E_{\pi_\beta}[v] = \mathrm{Cov}_{\pi_\beta}(u, v)$, where $u$ is the
overseer's reward and $v$ is hidden ground truth. Training against an overseer therefore helps
exactly as long as reward and ground truth covary positively under the *current* policy. When the
covariance turns negative, gaming has started. We can estimate that covariance (a) before training, from
sampled pools, exact best-of-$N$/tilting curves on game trees and prompted adversarial agents, and
(b) during training, from a small audited sample. The project tests whether (a) forecasts (b).

**Experiments on Tinker** (Inkling-Small, Inkling and one smaller open model as a scale axis):
1. *Gaming onset under RL.* RL (LoRA, GRPO) against four overseers: a single LLM judge, debate,
   a judge with random audits, and a CoT/action monitor. The domains have measured ground truth: code
   with hidden tests, text-to-SQL over a private database, and agentic work in a simulated company
   with hidden records. At every checkpoint we log reward and ground truth, and we map the step at
   which gaming starts against KL budget and model size.
2. *Forecasting from cheap runs.* Before each run we pre-register forecasts of gaming onset and final
   ground truth. The forecasts come from pre-training measurements and from short runs on smaller
   models. We score them against the full runs and report where the extrapolation breaks.
3. *Detection and transfer.* Do signals inside the mechanism (judge uncertainty, debater
   disagreement, failed verifications, monitor scores) flag gamed episodes (AUROC against ground
   truth)? Does a strategy learned against one overseer or domain carry over to others, e.g. to a
   CoT monitor the policy never trained against?

**Construct validity.** Ground truth is *measured* (hidden tests, execution on private data, the
environment's own trusted records), never taken from instructions or from anything the agent can
write, and it is never shown to the mechanism. Agent code runs sandboxed. Items are checked for blind
solvability, and we report judge parse failures. A claim only counts as verified when a trusted
checker confirms it. Forecasts are registered before the runs.

**Feasibility.** The OversightArena library already exists. It has mechanisms and reward rules, the
three domains, stateful forkable environments, RL environments and reward functions, and exact
best-of-$N$ on game trees, backed by several hundred offline tests. It works with Inspect and ControlArena. The
grant pays for the missing piece: real RL at several scales. Month 1: pilot runs, cost calibration and
registered forecasts. Months 2–4: the main grid. Months 5–6: transfer and detection, paper and
release. About 60% of credits go to the main grid, 20% to small forecasting runs, and 20% to
transfer and ablations.

**Simplicity and generality.** An overseer is a short Python function plus a reward rule, so any
monitor, judge or control protocol can be tested this way. The main output per mechanism is one
curve (reward vs. ground truth over training) and one number (steps until it is gamed). This applies
to open- and closed-weight training alike.

**Outputs.** A paper; open environments and reward functions; released episode logs of gamed vs.
honest behaviour for training detectors; a public table of which oversight mechanisms stay
ungamed for how long.

**Team.** [PI: one line on relevant prior work] · [contributors] · CVs attached.
