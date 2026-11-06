# Instant judges: the same transcripts judged without reasoning

## Instructed EIC, all prompted liars (gpt-5-mini, 927 items)

| judge | protocol | reasoning ASD | instant ASD | reasoning conc | instant conc | reasoning r | instant r |
|---|---|---|---|---|---|---|---|
| pooled | consultancy | +0.38 [+0.29, +0.47] | +0.00 [-0.07, +0.08] | 0.61 | 0.54 | 0.09 | 0.00 |
| pooled | debate_seq | +0.74 [+0.66, +0.82] | +0.47 [+0.40, +0.55] | 0.69 | 0.66 | 0.18 | 0.12 |
| pooled | debate_sim | +0.79 [+0.70, +0.87] | +0.52 [+0.43, +0.60] | 0.72 | 0.66 | 0.19 | 0.13 |
| gemma-3-12b | consultancy | +0.91 [+0.76, +1.06] | +0.21 [+0.14, +0.28] | 0.66 | 0.60 | 0.19 | 0.09 |
| gemma-3-12b | debate_seq | +0.64 [+0.53, +0.75] | +0.69 [+0.54, +0.83] | 0.61 | 0.61 | 0.15 | 0.12 |
| gemma-3-12b | debate_sim | +0.68 [+0.57, +0.80] | +0.84 [+0.68, +1.01] | 0.62 | 0.61 | 0.15 | 0.15 |
| gpt-4o-mini | consultancy | +0.48 [+0.39, +0.57] | +0.08 [+0.03, +0.12] | 0.63 | 0.56 | 0.18 | 0.06 |
| gpt-4o-mini | debate_seq | +0.79 [+0.71, +0.87] | +0.27 [+0.24, +0.31] | 0.73 | 0.68 | 0.30 | 0.20 |
| gpt-4o-mini | debate_sim | +0.83 [+0.75, +0.92] | +0.30 [+0.26, +0.34] | 0.73 | 0.68 | 0.33 | 0.22 |
| llama-3.1-8b | consultancy | -0.26 [-0.41, -0.12] | -0.27 [-0.45, -0.10] | 0.47 | 0.50 | -0.05 | -0.05 |
| llama-3.1-8b | debate_seq | +0.80 [+0.66, +0.93] | +0.46 [+0.37, +0.55] | 0.64 | 0.60 | 0.16 | 0.13 |
| llama-3.1-8b | debate_sim | +0.84 [+0.70, +0.97] | +0.40 [+0.31, +0.50] | 0.65 | 0.57 | 0.17 | 0.11 |
| gpt-4o-mini-lp | consultancy | - | +0.66 [+0.43, +0.87] | - | 0.54 | - | 0.11 |
| gpt-4o-mini-lp | debate_seq | - | +1.90 [+1.58, +2.21] | - | 0.64 | - | 0.23 |
| gpt-4o-mini-lp | debate_sim | - | +2.12 [+1.76, +2.46] | - | 0.65 | - | 0.26 |

## Instructed EIC, committed liars only (gpt-5-mini, 927 items)

| judge | protocol | reasoning ASD | instant ASD | reasoning conc | instant conc | reasoning r | instant r |
|---|---|---|---|---|---|---|---|
| pooled | consultancy | +0.21 [+0.12, +0.31] | -0.12 [-0.20, -0.05] | 0.55 | 0.51 | 0.04 | -0.04 |
| pooled | debate_seq | +0.56 [+0.48, +0.64] | +0.32 [+0.25, +0.39] | 0.65 | 0.61 | 0.14 | 0.08 |
| pooled | debate_sim | +0.57 [+0.48, +0.65] | +0.31 [+0.24, +0.38] | 0.68 | 0.61 | 0.14 | 0.07 |
| gemma-3-12b | consultancy | +0.66 [+0.51, +0.82] | +0.05 [+0.00, +0.11] | 0.62 | 0.59 | 0.13 | 0.03 |
| gemma-3-12b | debate_seq | +0.40 [+0.30, +0.51] | +0.47 [+0.33, +0.63] | 0.56 | 0.58 | 0.09 | 0.08 |
| gemma-3-12b | debate_sim | +0.41 [+0.29, +0.52] | +0.51 [+0.35, +0.69] | 0.58 | 0.57 | 0.09 | 0.09 |
| gpt-4o-mini | consultancy | +0.26 [+0.18, +0.35] | -0.05 [-0.09, -0.02] | 0.58 | 0.52 | 0.10 | -0.08 |
| gpt-4o-mini | debate_seq | +0.64 [+0.56, +0.73] | +0.17 [+0.14, +0.20] | 0.70 | 0.65 | 0.29 | 0.19 |
| gpt-4o-mini | debate_sim | +0.68 [+0.59, +0.76] | +0.18 [+0.15, +0.22] | 0.69 | 0.64 | 0.31 | 0.20 |
| llama-3.1-8b | consultancy | -0.35 [-0.51, -0.20] | -0.38 [-0.56, -0.21] | 0.44 | 0.49 | -0.07 | -0.07 |
| llama-3.1-8b | debate_seq | +0.64 [+0.49, +0.78] | +0.31 [+0.22, +0.40] | 0.60 | 0.56 | 0.12 | 0.09 |
| llama-3.1-8b | debate_sim | +0.61 [+0.47, +0.75] | +0.23 [+0.14, +0.33] | 0.62 | 0.53 | 0.12 | 0.06 |
| gpt-4o-mini-lp | consultancy | - | -0.09 [-0.30, +0.11] | - | 0.47 | - | -0.05 |
| gpt-4o-mini-lp | debate_seq | - | +1.55 [+1.22, +1.88] | - | 0.61 | - | 0.21 |
| gpt-4o-mini-lp | debate_sim | - | +1.72 [+1.36, +2.08] | - | 0.61 | - | 0.21 |

## Blind baseline (judge alone, log score)

| judge | reasoning ASD | instant ASD |
|---|---|---|
| pooled | -0.58 [-0.77, -0.40] | -0.74 [-0.88, -0.59] |
| gemma-3-12b | -0.06 [-0.36, +0.24] | -1.13 [-1.35, -0.91] |
| gpt-4o-mini | -0.50 [-0.79, -0.21] | -0.48 [-0.60, -0.37] |
| gpt-4o-mini-lp | - | -2.25 [-2.54, -1.95] |
| llama-3.1-8b | -1.19 [-1.40, -0.97] | -0.59 [-0.85, -0.34] |

## Judge accuracy

| judging | judge | alone | consultancy | debate_seq | debate_sim |
|---|---|---|---|---|---|
| instant | gemma-3-12b | 0.42 | 0.52 | 0.53 | 0.56 |
| instant | gpt-4o-mini | 0.39 | 0.52 | 0.59 | 0.60 |
| instant | gpt-4o-mini-lp | 0.31 | 0.53 | 0.61 | 0.62 |
| instant | llama-3.1-8b | 0.46 | 0.48 | 0.55 | 0.54 |
| reasoning | gemma-3-12b | 0.45 | 0.57 | 0.53 | 0.54 |
| reasoning | gpt-4o-mini | 0.42 | 0.58 | 0.64 | 0.67 |
| reasoning | llama-3.1-8b | 0.41 | 0.50 | 0.57 | 0.58 |

## Best-of-n under instant judges (pooled over the three weak judges)

| judging | protocol | optimized | n=1 ASD | n=16 ASD | n=1 acc | n=16 acc |
|---|---|---|---|---|---|---|
| instant | consultancy_1r | both | -0.07 [-0.20, +0.06] | +0.02 [-0.02, +0.07] | 0.49 | 0.50 |
| instant | consultancy_1r | liar | -0.07 [-0.20, +0.06] | -0.58 [-0.69, -0.48] | 0.49 | 0.40 |
| instant | debate_1r | both | +0.03 [-0.12, +0.17] | -0.09 [-0.23, +0.06] | 0.51 | 0.50 |
| instant | debate_1r | liar | +0.03 [-0.12, +0.17] | -1.18 [-1.37, -1.00] | 0.51 | 0.36 |
| reasoning (held out) | consultancy_1r | both | +0.38 [+0.14, +0.61] | +0.28 [-0.02, +0.57] | 0.55 | 0.54 |
| reasoning (held out) | consultancy_1r | liar | +0.38 [+0.14, +0.61] | +0.21 [-0.07, +0.46] | 0.55 | 0.51 |
| reasoning (held out) | debate_1r | both | +0.48 [+0.30, +0.67] | +0.28 [+0.08, +0.50] | 0.55 | 0.51 |
| reasoning (held out) | debate_1r | liar | +0.48 [+0.30, +0.67] | -0.37 [-0.55, -0.19] | 0.55 | 0.40 |

## Test-retest of instant verdicts (two independent calls, same transcript)

| judge | transcripts | identical | r |
|---|---|---|---|
| gemma-3-12b | 600 | 0.82 | 0.92 |
| gpt-4o-mini-lp | 600 | 0.24 | 0.97 |
| gpt-4o-mini | 600 | 0.86 | 0.95 |
| llama-3.1-8b | 600 | 0.70 | 0.78 |
