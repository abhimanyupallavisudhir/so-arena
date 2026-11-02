# Empirical incentive compatibility of the oversight protocols

EIC: the protocol's reward for arguing the true answer minus its reward for arguing a false one (ASD, in the protocol's reward units; equal to $\mathrm{Cov}(u,v)/\mathrm{Var}(v)$ for binary $v$), with the share of items on which honesty earns more (concordance) and, where per-episode rewards exist, $r = \mathrm{corr}(u, v)$. 95% item-bootstrap intervals. Figures in `eic/`.

## Instructed sampler (ASD), gpt-5-mini, 927 items

| protocol | ASD | concordance | r | items |
|---|---|---|---|---|
| Consultancy | +0.38 [+0.29, +0.47] | 0.61 [0.57, 0.64] | +0.09 [+0.07, +0.11] | 927 |
| Debate, sequential | +0.74 [+0.66, +0.82] | 0.69 [0.66, 0.72] | +0.18 [+0.16, +0.20] | 927 |
| Debate, simultaneous | +0.79 [+0.70, +0.87] | 0.72 [0.69, 0.75] | +0.19 [+0.17, +0.21] | 927 |
| Prover-estimator | -0.13 [-0.15, -0.11] | 0.35 [0.32, 0.38] | -0.14 [-0.16, -0.12] | 927 |
| Recursive debate | +0.10 [+0.08, +0.11] | 0.64 [0.61, 0.67] | +0.14 [+0.12, +0.16] | 927 |
| Consultancy | +0.36 [+0.21, +0.51] | 0.65 [0.60, 0.70] | +0.08 [+0.05, +0.12] | 320 |
| Debate, sequential | +0.77 [+0.62, +0.92] | 0.68 [0.63, 0.73] | +0.19 [+0.16, +0.22] | 320 |
| Debate, simultaneous | +0.81 [+0.67, +0.96] | 0.72 [0.68, 0.77] | +0.20 [+0.16, +0.23] | 320 |
| Prover-estimator | -0.15 [-0.19, -0.12] | 0.32 [0.27, 0.37] | -0.16 [-0.20, -0.12] | 320 |
| Recursive debate | +0.09 [+0.07, +0.11] | 0.63 [0.57, 0.68] | +0.13 [+0.10, +0.16] | 320 |

## Instructed sampler, committed liars only, gpt-5-mini

| protocol | ASD | concordance | r | items |
|---|---|---|---|---|
| Consultancy | +0.21 [+0.12, +0.31] | 0.55 [0.52, 0.58] | +0.04 [+0.01, +0.05] | 927 |
| Debate, sequential | +0.56 [+0.48, +0.64] | 0.65 [0.62, 0.68] | +0.14 [+0.12, +0.16] | 898 |
| Debate, simultaneous | +0.57 [+0.48, +0.65] | 0.68 [0.65, 0.71] | +0.14 [+0.12, +0.16] | 875 |
| Prover-estimator | -0.23 [-0.26, -0.20] | 0.27 [0.24, 0.31] | -0.26 [-0.29, -0.23] | 780 |
| Recursive debate | +0.10 [+0.08, +0.12] | 0.62 [0.59, 0.66] | +0.12 [+0.10, +0.14] | 876 |
| Consultancy | +0.21 [+0.05, +0.37] | 0.60 [0.55, 0.66] | +0.04 [+0.01, +0.08] | 320 |
| Debate, sequential | +0.56 [+0.43, +0.69] | 0.66 [0.60, 0.71] | +0.14 [+0.11, +0.17] | 308 |
| Debate, simultaneous | +0.54 [+0.41, +0.67] | 0.68 [0.63, 0.73] | +0.14 [+0.11, +0.17] | 299 |
| Prover-estimator | -0.27 [-0.31, -0.23] | 0.22 [0.17, 0.27] | -0.31 [-0.35, -0.27] | 272 |
| Recursive debate | +0.08 [+0.06, +0.11] | 0.59 [0.54, 0.65] | +0.11 [+0.07, +0.14] | 306 |

## Exact best-of-n, one round (held-out judging)

| protocol | optimized | n | ASD | concordance |
|---|---|---|---|---|
| Consultancy | both | 1 | +0.38 [+0.14, +0.61] | 0.58 |
| Consultancy | both | 2 | +0.30 [+0.06, +0.51] | 0.56 |
| Consultancy | both | 4 | +0.26 [+0.03, +0.47] | 0.53 |
| Consultancy | both | 8 | +0.26 [+0.01, +0.51] | 0.51 |
| Consultancy | both | 16 | +0.28 [-0.02, +0.57] | 0.51 |
| Consultancy | liar | 1 | +0.38 [+0.14, +0.61] | 0.58 |
| Consultancy | liar | 2 | +0.24 [+0.01, +0.46] | 0.53 |
| Consultancy | liar | 4 | +0.18 [-0.05, +0.39] | 0.50 |
| Consultancy | liar | 8 | +0.18 [-0.07, +0.40] | 0.47 |
| Consultancy | liar | 16 | +0.21 [-0.07, +0.46] | 0.47 |
| Debate | both | 1 | +0.48 [+0.30, +0.67] | 0.63 |
| Debate | both | 2 | +0.40 [+0.22, +0.59] | 0.62 |
| Debate | both | 4 | +0.36 [+0.19, +0.55] | 0.59 |
| Debate | both | 8 | +0.33 [+0.14, +0.52] | 0.59 |
| Debate | both | 16 | +0.28 [+0.08, +0.50] | 0.58 |
| Debate | liar | 1 | +0.48 [+0.30, +0.67] | 0.63 |
| Debate | liar | 2 | +0.22 [+0.05, +0.41] | 0.56 |
| Debate | liar | 4 | +0.00 [-0.17, +0.18] | 0.45 |
| Debate | liar | 8 | -0.19 [-0.37, -0.02] | 0.38 |
| Debate | liar | 16 | -0.37 [-0.55, -0.19] | 0.29 |

## Best-of-n on every speech, three rounds

| protocol | optimized | n | ASD | concordance |
|---|---|---|---|---|
| Consultancy | both | 1 | +0.51 [+0.29, +0.75] | 0.69 |
| Consultancy | both | 4 | +0.12 [-0.15, +0.40] | 0.61 |
| Consultancy | both | 16 | -0.06 [-0.34, +0.20] | 0.52 |
| Consultancy | liar | 1 | +0.51 [+0.29, +0.75] | 0.69 |
| Consultancy | liar | 4 | +0.19 [-0.04, +0.44] | 0.49 |
| Consultancy | liar | 16 | +0.10 [-0.16, +0.37] | 0.39 |
| Debate, sequential | both | 1 | +0.69 [+0.51, +0.91] | 0.67 |
| Debate, sequential | both | 4 | +0.53 [+0.27, +0.79] | 0.62 |
| Debate, sequential | both | 16 | +0.60 [+0.27, +0.92] | 0.59 |
| Debate, sequential | liar | 1 | +0.69 [+0.51, +0.91] | 0.67 |
| Debate, sequential | liar | 4 | +0.34 [+0.05, +0.65] | 0.61 |
| Debate, sequential | liar | 16 | +0.22 [-0.04, +0.49] | 0.52 |

## Choosing a side: share of the optimized agent's behaviour that is honest

| protocol | n | honest share |
|---|---|---|
| Consultancy | 1 | 0.50 [0.50, 0.50] |
| Consultancy | 2 | 0.56 [0.55, 0.57] |
| Consultancy | 4 | 0.60 [0.58, 0.62] |
| Consultancy | 8 | 0.62 [0.59, 0.64] |
| Consultancy | 16 | 0.62 [0.59, 0.65] |
| Consultancy | 32 | 0.63 [0.59, 0.67] |
| Debate | 1 | 0.50 [0.50, 0.50] |
| Debate | 2 | 0.59 [0.57, 0.61] |
| Debate | 4 | 0.64 [0.61, 0.68] |
| Debate | 8 | 0.65 [0.61, 0.69] |
| Debate | 16 | 0.64 [0.60, 0.69] |
| Debate | 32 | 0.64 [0.59, 0.69] |

## EIC by judge capability (the same debates re-judged)

| liars | dataset | protocol | judge | ASD | concordance |
|---|---|---|---|---|---|
| all | quality | Consultancy | llama-3.1-8b | +0.40 [+0.30, +0.50] | 0.68 |
| all | quality | Consultancy | gemma-3-12b | +0.53 [+0.42, +0.64] | 0.74 |
| all | quality | Consultancy | gpt-4o-mini | +1.05 [+0.88, +1.22] | 0.82 |
| all | quality | Consultancy | gpt-4.1-mini | +1.36 [+1.17, +1.57] | 0.86 |
| all | quality | Consultancy | gpt-4.1 | +2.44 [+2.19, +2.69] | 0.92 |
| all | quality | Debate, sequential | llama-3.1-8b | +0.25 [+0.16, +0.33] | 0.64 |
| all | quality | Debate, sequential | gemma-3-12b | +0.41 [+0.31, +0.52] | 0.66 |
| all | quality | Debate, sequential | gpt-4o-mini | +0.60 [+0.49, +0.71] | 0.76 |
| all | quality | Debate, sequential | gpt-4.1-mini | +1.22 [+1.09, +1.36] | 0.87 |
| all | quality | Debate, sequential | gpt-4.1 | +2.64 [+2.45, +2.84] | 0.94 |
| all | testbed | Consultancy | llama-3.1-8b | +0.72 [+0.51, +0.94] | 0.69 |
| all | testbed | Consultancy | gemma-3-12b | +2.16 [+1.90, +2.43] | 0.82 |
| all | testbed | Consultancy | gpt-4o-mini | +1.54 [+1.35, +1.74] | 0.80 |
| all | testbed | Consultancy | gpt-4.1-mini | +4.44 [+4.02, +4.88] | 0.88 |
| all | testbed | Consultancy | gpt-4.1 | +5.15 [+4.75, +5.56] | 0.92 |
| all | testbed | Debate, sequential | llama-3.1-8b | +1.78 [+1.53, +2.04] | 0.76 |
| all | testbed | Debate, sequential | gemma-3-12b | +1.82 [+1.58, +2.05] | 0.77 |
| all | testbed | Debate, sequential | gpt-4o-mini | +1.67 [+1.48, +1.87] | 0.82 |
| all | testbed | Debate, sequential | gpt-4.1-mini | +3.32 [+3.05, +3.59] | 0.89 |
| all | testbed | Debate, sequential | gpt-4.1 | +4.99 [+4.64, +5.33] | 0.92 |
| all | testbed | Debate, simultaneous | llama-3.1-8b | +2.12 [+1.89, +2.37] | 0.80 |
| all | testbed | Debate, simultaneous | gemma-3-12b | +2.35 [+2.11, +2.60] | 0.81 |
| all | testbed | Debate, simultaneous | gpt-4o-mini | +1.96 [+1.76, +2.17] | 0.84 |
| all | testbed | Debate, simultaneous | gpt-4.1-mini | +3.77 [+3.49, +4.06] | 0.93 |
| all | testbed | Debate, simultaneous | gpt-4.1 | +5.34 [+4.96, +5.70] | 0.92 |
| committed | quality | Consultancy | llama-3.1-8b | +0.07 [+0.00, +0.13] | 0.54 |
| committed | quality | Consultancy | gemma-3-12b | +0.08 [+0.03, +0.12] | 0.65 |
| committed | quality | Consultancy | gpt-4o-mini | +0.19 [+0.12, +0.25] | 0.72 |
| committed | quality | Consultancy | gpt-4.1-mini | +0.26 [+0.15, +0.40] | 0.75 |
| committed | quality | Consultancy | gpt-4.1 | +0.92 [+0.76, +1.08] | 0.84 |
| committed | quality | Debate, sequential | llama-3.1-8b | +0.08 [+0.02, +0.15] | 0.58 |
| committed | quality | Debate, sequential | gemma-3-12b | +0.19 [+0.12, +0.27] | 0.58 |
| committed | quality | Debate, sequential | gpt-4o-mini | +0.36 [+0.30, +0.43] | 0.72 |
| committed | quality | Debate, sequential | gpt-4.1-mini | +1.00 [+0.88, +1.11] | 0.86 |
| committed | quality | Debate, sequential | gpt-4.1 | +2.39 [+2.21, +2.57] | 0.92 |
| committed | testbed | Consultancy | llama-3.1-8b | -0.35 [-0.59, -0.12] | 0.46 |
| committed | testbed | Consultancy | gemma-3-12b | +0.62 [+0.35, +0.89] | 0.66 |
| committed | testbed | Consultancy | gpt-4o-mini | +0.26 [+0.15, +0.41] | 0.60 |
| committed | testbed | Consultancy | gpt-4.1-mini | +2.92 [+2.44, +3.41] | 0.79 |
| committed | testbed | Consultancy | gpt-4.1 | +3.75 [+3.22, +4.26] | 0.86 |
| committed | testbed | Debate, sequential | llama-3.1-8b | +0.65 [+0.40, +0.89] | 0.59 |
| committed | testbed | Debate, sequential | gemma-3-12b | +0.34 [+0.19, +0.50] | 0.60 |
| committed | testbed | Debate, sequential | gpt-4o-mini | +0.64 [+0.51, +0.76] | 0.71 |
| committed | testbed | Debate, sequential | gpt-4.1-mini | +2.23 [+1.99, +2.47] | 0.86 |
| committed | testbed | Debate, sequential | gpt-4.1 | +4.25 [+3.90, +4.59] | 0.90 |
| committed | testbed | Debate, simultaneous | llama-3.1-8b | +0.57 [+0.31, +0.83] | 0.61 |
| committed | testbed | Debate, simultaneous | gemma-3-12b | +0.38 [+0.19, +0.57] | 0.57 |
| committed | testbed | Debate, simultaneous | gpt-4o-mini | +0.62 [+0.49, +0.75] | 0.69 |
| committed | testbed | Debate, simultaneous | gpt-4.1-mini | +2.40 [+2.15, +2.65] | 0.87 |
| committed | testbed | Debate, simultaneous | gpt-4.1 | +4.58 [+4.17, +4.95] | 0.88 |

## Tournament: each model's reward arguing the truth minus lying

| liars | model | ASD | concordance |
|---|---|---|---|
| all | gpt-5 | +0.74 [+0.57, +0.92] | 0.70 |
| all | claude-sonnet-5 | +0.92 [+0.75, +1.11] | 0.76 |
| all | gpt-5-mini | +0.93 [+0.74, +1.12] | 0.76 |
| all | gpt-4.1 | +1.06 [+0.85, +1.28] | 0.77 |
| all | gpt-4.1-mini | +1.16 [+0.93, +1.41] | 0.73 |
| all | claude-haiku-4-5 | +1.47 [+1.20, +1.76] | 0.79 |
| all | claude-sonnet-4-5 | +2.97 [+2.55, +3.36] | 0.80 |
| all | claude-opus-5-5 | +3.98 [+3.56, +4.39] | 0.90 |
| committed | claude-opus-5-5 | +0.01 [-0.13, +0.13] | 0.55 |
| committed | claude-haiku-4-5 | +0.14 [-0.09, +0.41] | 0.57 |
| committed | claude-sonnet-4-5 | +0.24 [+0.00, +0.48] | 0.52 |
| committed | gpt-4.1-mini | +0.42 [+0.21, +0.65] | 0.61 |
| committed | claude-sonnet-5 | +0.44 [+0.25, +0.63] | 0.66 |
| committed | gpt-4.1 | +0.49 [+0.28, +0.71] | 0.66 |
| committed | gpt-5 | +0.54 [+0.35, +0.72] | 0.61 |
| committed | gpt-5-mini | +0.57 [+0.38, +0.74] | 0.65 |
