# The conceptual-reasoning testbed

Conceptual questions (decision theory, ethics, AI risk) have no answer key. A method for producing or judging
conceptual arguments - debate, a trained judge, essays scored by their downstream use - therefore can't be
checked against the truth on the questions we care about. The testbed supplies questions that share the
structure: the answer has to be argued for, because it can't be looked up or cheaply worked out in one's
head. The difference is that the experimenter knows the answer, by computation, proof or experiment, and keeps
it hidden from the method under test.

It serves three uses:

1. **Comparing oversight mechanisms.** Run a protocol on testbed items with the answer hidden, and score its
   verdicts. See [`experiments/oversight_testbed.py`](../experiments/oversight_testbed.py).
2. **Validating proxies.** Do arguments that an LLM judge, or an expert, rates highly actually lead to the
   right answer?
3. **Supplying indicators for essays-as-latents.** Each *family* is a set of questions sharing one latent
   structure. An essay about the family is scored by how it changes a frozen reader's predictions of the
   family's items. This lives in the `lvpm` repository.

The caveat throughout: testbed questions are crisper than real philosophy. Doing well on the testbed is
necessary for a method, not sufficient.

## Loading

```python
from so_arena.domains import get_domain

get_domain("testbed").load()                                            # the default tier-A families, 40 items each
get_domain("testbed", families=["decisions", "rlhack/sensor"], per_family=60).load()
get_domain("decisions", families=["newcomb"]).families()["decisions/newcomb"].card   # what a writer may read
```

Every family follows the same conventions (`so_arena.domains.testbed.base`):

- Items have two answers, `A` and `B`, worth $+1$ and $-1$.
- The true answer is at position `A` on exactly half of each family's items.
- The generator's "canonical" answer is true on half of the items, by rejection sampling: one-boxing wins
  half the Newcomb items, exploitation happens in half the reward-hacking experiments, and so on. No single
  slogan answers a family. The exception is `numbers/prime_race`, whose natural base rate is the point.
- `metadata` holds `family`, `tier` (`A` for objective ground truth, `B` for expert judgment), `split`
  (`train`/`test`, a stable hash of the item id) and the generating parameters.
- Ground-truth quantities (expected payoffs, counts, SAT results, measured outcomes) are in
  `ground_truth.data`. That field is stripped before any mechanism sees the item.

## Families

| family | question | ground truth | checked by |
|---|---|---|---|
| `decisions/newcomb` | two-box problems where the predictor reads the program with probability $r$ (accuracy $q$), otherwise guesses from a base rate | closed-form ex-ante expected payoff | Monte Carlo simulation of the described process |
| `decisions/lesion` | smoking lesion: population correlation through a gene; X's program is set independently of its genes | " | " |
| `decisions/twin` | prisoner's dilemma against a copy with probability $\rho$ | " | " |
| `decisions/mugging` | counterfactual mugging, ex ante | " | " |
| `decisions/beauty` | Sleeping-Beauty bets, every bet settled vs only one | " | " |
| `decisions/driver` | absent-minded driver: which exit probability pays more | " | " |
| `decisions/damascus` | Death in Damascus: staying, travelling, or randomizing | " | " |
| `decisions/hitchhiker` | Parfit's hitchhiker | " | " |
| `socialchoice/rankings` | can a social welfare function satisfy these axioms (2–3 voters, 3 alternatives, full or single-peaked domain, strict or weak output)? | SAT over all rules (CaDiCaL) | reproduces Arrow, the dictatorship and single-peaked majority cases |
| `socialchoice/winners` | the same for rules that pick one winner (strategy-proofness, Maskin monotonicity, Condorcet, anonymity, neutrality, ...) | SAT | reproduces Gibbard–Satterthwaite, Muller–Satterthwaite, Moulin, the median rule |
| `numbers/prime_race` | more primes $\le N$ in class $r_1$ or $r_2$ mod $q$? | sieve | brute force |
| `numbers/counts` | is the count of twin, $(p,p+6)$ or Sophie Germain primes, of primes $n^2+1$, of Goldbach representations, or of sums of two squares above a threshold placed 4–30% from the truth? | exact count | brute force |
| `numbers/collatz` | does more than a share $f$ of starts escape under $n \mapsto n/2$ or $an+b$? | iteration | Python big integers |
| `rlhack/sensor` | does tabular Q-learning farm a repeatable reward sensor (majority of 6 runs)? | **measured**: 6 training runs | re-runs reproduce the records |
| `rlhack/fake_goal` | does it go to a fake terminal goal? | measured | " |
| `rlhack/watched_sensor` | does it farm a sensor that an overseer sometimes catches and punishes? | measured | " |
| `agents/auction` | which of two conditions gives the auctioneer more revenue from LLM bidders? Conditions vary model, number of bidders, first vs second price, chat, bid revelation | **measured**: 3 sessions × 6 rounds per condition | |
| `agents/sycophancy` | which condition gives higher GSM8K accuracy when the user suggests a wrong answer? Conditions vary model, phrasing, show-work | measured: 30 problems per condition | |
| `agents/persona` | which condition gives more EDT-agreeing answers on decision-theory attitude questions? Conditions vary model and persona (pirate, Stoic, Calvinist, "many copies of you", ...) | measured: 40 questions per condition | |
| `dtbench/capabilities` | the 407 expert-written DTBench questions (340 after dropping trivia and rote-memorization items), as two-option items | expert answers (tier A-like) | |
| `dtbench/attitudes` | the 130 attitude questions, with EDT/CDT labels; used as a behavioural battery | none (tier B) | |

Notes on individual families:

- **Decision problems.** The question asks which *installed program* gives the higher *ex-ante* expected
  payoff. That makes the answer objective while keeping the dispute: dominance arguments, "the boxes are
  already filled", and population statistics all suggest the wrong answer on some items. Every item includes
  one such true but misleading statement.
- **Social choice.** Theorem hypotheses (voters, alternatives, domain, resoluteness) are varied, so
  recalling Arrow is not enough.
- **Number theory.** The intended route is heuristic: Chebyshev's bias, Hardy–Littlewood with the singular
  series, Landau–Ramanujan, and the drift $\log a - 2\log 2$ for Collatz-like maps.
- **RL reward hacking.** Only clear-cut experiments are kept: at most 1, or at least 5, of the 6 runs exploit.
  That selection is part of the family's definition. A 12-feature logistic regression on the parameters
  predicts outcomes with cross-validated accuracy 0.90 (sensor), 0.94 (fake goal) and 0.87 (watched). So the
  outcomes follow learnable structure that a good theory should capture. Reading the map and the training
  settings is the reasoner's job.
- **LLM-agent experiments.** Each condition is measured once. Items are pairs of conditions differing in
  exactly one factor, kept only when the metrics differ by at least `MIN_GAP` (0.08 revenue ratio, 0.10
  accuracy or EDT share). Which condition is listed first is a coin flip.

Data handling:

- Decision, social-choice and number-theory items are generated on load, deterministically in the seed.
- RL and LLM-agent outcomes are experiment results. They are committed as
  `src/so_arena/data/samples/testbed_{rlhack,agents}.jsonl`, with every condition's parameters, measurement and
  cost. Regenerate them with `scripts/build_testbed.py`. Building all the agent experiments cost \$3.00 (one
  build plus a rebuild of two families after a parsing fix).
- DTBench is downloaded on first use from its public, password-protected archive into the dataset cache. It is
  never committed; its authors ask that it not be republished in plain text.

## Not yet included

- **LMCA (tier B).** Access to the rated-arguments dataset is by request form. Once it is available, a loader
  turns each (position, critique, expert rating) into an item, with the rating as the graded value.
- **Research-outcome prediction** (Wen et al. 2025, idea pairs with benchmark outcomes). The data release
  could not be located; the paper's pipeline could be re-run on post-cutoff safety papers.
- **Fine-tuning experiments** (e.g. "does training on pirate speech change decision-theoretic attitudes?").
  These would use a provider fine-tuning API: a few dollars and 10–30 minutes per run. The `agents/persona`
  family is the prompting analogue.
- **Population-ethics axiom consistency.** The same SAT approach as social choice, over orderings of finite
  sets of populations.
