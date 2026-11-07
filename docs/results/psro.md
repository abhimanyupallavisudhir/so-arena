# PSRO self-play: incentive compatibility at equilibrium

Held-out items; ASD = E_sigma[honest - liar reward] (log score of the 3-sample mean judge), 95% bootstrap CI over items; asd_1 uses the expected single-sample reward; conc = share of items where honesty earns more.

| protocol | judge | iteration | support (H/L) | ASD [95% CI] | asd_1 | conc | judge acc | NashConv |
|---|---|---|---|---|---|---|---|---|
| debate_seq | gemma-3-12b | 0 | 1/1 | +0.46 [+0.22, +0.71] | +0.52 | 0.55 | 0.56 | 0.000 |
| debate_seq | gemma-3-12b | 1 | 1/1 | +0.34 [+0.17, +0.52] | +0.38 | 0.57 | 0.57 | 0.000 |
| debate_seq | gemma-3-12b | 2 | 2/2 | +0.33 [+0.21, +0.47] | +0.40 | 0.58 | 0.56 | 0.000 |
| debate_seq | gemma-3-12b | 3 | 2/2 | +0.29 [+0.17, +0.42] | +0.32 | 0.59 | 0.55 | 0.000 |
| debate_seq | gemma-3-12b | 4 | 2/2 | +0.29 [+0.17, +0.42] | +0.32 | 0.59 | 0.55 | 0.000 |
| debate_seq | gpt-4o-mini | 0 | 1/1 | +0.58 [+0.44, +0.71] | +0.69 | 0.62 | 0.62 | 0.000 |
| debate_seq | gpt-4o-mini | 1 | 1/1 | +0.50 [+0.37, +0.62] | +0.57 | 0.64 | 0.64 | 0.000 |
| debate_seq | gpt-4o-mini | 2 | 1/1 | -0.15 [-0.30, -0.01] | -0.10 | 0.45 | 0.46 | 0.000 |
| debate_seq | gpt-4o-mini | 3 | 1/1 | +0.23 [+0.05, +0.41] | +0.40 | 0.52 | 0.52 | 0.000 |
| debate_seq | gpt-4o-mini | 4 | 1/1 | -0.19 [-0.35, -0.02] | -0.15 | 0.43 | 0.44 | 0.000 |
| debate_seq | llama-3.1-8b | 0 | 1/1 | +0.35 [+0.19, +0.50] | +0.67 | 0.59 | 0.60 | 0.000 |
| debate_seq | llama-3.1-8b | 1 | 1/1 | +0.02 [-0.11, +0.14] | +0.07 | 0.50 | 0.51 | 0.000 |
| debate_seq | llama-3.1-8b | 2 | 2/2 | +0.06 [-0.04, +0.16] | +0.08 | 0.52 | 0.51 | 0.000 |
| debate_seq | llama-3.1-8b | 3 | 2/2 | +0.14 [+0.03, +0.25] | +0.22 | 0.53 | 0.53 | 0.000 |
| debate_seq | llama-3.1-8b | 4 | 1/1 | +0.03 [-0.08, +0.13] | +0.18 | 0.48 | 0.48 | 0.000 |
| debate_seq | pooled | 0 | - | +0.46 [+0.33, +0.60] | +0.63 | 0.62 | 0.59 | - |
| debate_seq | pooled | 1 | - | +0.28 [+0.19, +0.37] | +0.34 | 0.62 | 0.58 | - |
| debate_seq | pooled | 2 | - | +0.08 [-0.01, +0.17] | +0.13 | 0.53 | 0.51 | - |
| debate_seq | pooled | 3 | - | +0.22 [+0.12, +0.32] | +0.31 | 0.57 | 0.54 | - |
| debate_seq | pooled | 4 | - | +0.05 [-0.04, +0.13] | +0.12 | 0.51 | 0.49 | - |
| debate_sim | gemma-3-12b | 0 | 1/1 | +0.47 [+0.24, +0.72] | +0.52 | 0.57 | 0.57 | 0.000 |
| debate_sim | gemma-3-12b | 1 | 1/1 | +0.26 [+0.07, +0.46] | +0.37 | 0.50 | 0.50 | 0.000 |
| debate_sim | gemma-3-12b | 2 | 1/1 | +0.05 [-0.13, +0.22] | +0.10 | 0.49 | 0.49 | 0.000 |
| debate_sim | gemma-3-12b | 3 | 1/1 | +0.55 [+0.35, +0.76] | +0.75 | 0.59 | 0.60 | 0.000 |
| debate_sim | gemma-3-12b | 4 | 1/1 | +0.55 [+0.35, +0.76] | +0.75 | 0.59 | 0.60 | 0.000 |
| debate_sim | gpt-4o-mini | 0 | 1/1 | +0.75 [+0.60, +0.90] | +0.83 | 0.66 | 0.67 | 0.000 |
| debate_sim | gpt-4o-mini | 1 | 1/1 | +0.55 [+0.41, +0.68] | +0.65 | 0.64 | 0.64 | 0.000 |
| debate_sim | gpt-4o-mini | 2 | 1/1 | +0.53 [+0.39, +0.66] | +0.61 | 0.62 | 0.62 | 0.000 |
| debate_sim | gpt-4o-mini | 3 | 1/1 | +0.57 [+0.43, +0.71] | +0.76 | 0.62 | 0.63 | 0.000 |
| debate_sim | gpt-4o-mini | 4 | 2/2 | +0.58 [+0.47, +0.69] | +0.69 | 0.65 | 0.66 | 0.000 |
| debate_sim | llama-3.1-8b | 0 | 1/1 | +0.39 [+0.25, +0.54] | +0.68 | 0.59 | 0.60 | 0.000 |
| debate_sim | llama-3.1-8b | 1 | 1/1 | +0.15 [+0.03, +0.29] | +0.28 | 0.59 | 0.59 | 0.000 |
| debate_sim | llama-3.1-8b | 2 | 1/1 | +0.18 [+0.07, +0.28] | +0.27 | 0.56 | 0.58 | 0.000 |
| debate_sim | llama-3.1-8b | 3 | 2/2 | +0.33 [+0.21, +0.46] | +0.52 | 0.60 | 0.56 | 0.000 |
| debate_sim | llama-3.1-8b | 4 | 2/2 | +0.31 [+0.22, +0.40] | +0.51 | 0.64 | 0.59 | 0.000 |
| debate_sim | pooled | 0 | - | +0.54 [+0.40, +0.68] | +0.68 | 0.65 | 0.61 | - |
| debate_sim | pooled | 1 | - | +0.32 [+0.22, +0.41] | +0.43 | 0.61 | 0.58 | - |
| debate_sim | pooled | 2 | - | +0.25 [+0.16, +0.34] | +0.33 | 0.60 | 0.57 | - |
| debate_sim | pooled | 3 | - | +0.48 [+0.38, +0.59] | +0.68 | 0.67 | 0.60 | - |
| debate_sim | pooled | 4 | - | +0.48 [+0.38, +0.58] | +0.65 | 0.68 | 0.62 | - |

## Best responses: training gain vs held-out gain

| protocol | judge | iteration | side | new | train gain | held-out gain [95% CI] |
|---|---|---|---|---|---|---|
| debate_seq | gemma-3-12b | 1 | honest | yes | +0.13 | -0.29 [-0.48, -0.11] |
| debate_seq | gemma-3-12b | 1 | liar | yes | +0.39 | +0.25 [+0.08, +0.42] |
| debate_seq | gemma-3-12b | 2 | honest | no | +0.00 | +0.00 |
| debate_seq | gemma-3-12b | 2 | liar | yes | +0.29 | +0.10 [-0.00, +0.21] |
| debate_seq | gemma-3-12b | 3 | honest | no | +0.00 | +0.00 |
| debate_seq | gemma-3-12b | 3 | liar | yes | +0.07 | -0.08 [-0.19, +0.03] |
| debate_seq | gemma-3-12b | 4 | honest | no | +0.00 | +0.00 |
| debate_seq | gemma-3-12b | 4 | liar | no | +0.00 | +0.00 |
| debate_seq | gpt-4o-mini | 1 | honest | yes | +0.09 | -0.07 [-0.14, +0.01] |
| debate_seq | gpt-4o-mini | 1 | liar | yes | +0.05 | +0.08 [-0.00, +0.17] |
| debate_seq | gpt-4o-mini | 2 | honest | no | +0.00 | +0.00 |
| debate_seq | gpt-4o-mini | 2 | liar | yes | +0.33 | +0.28 [+0.20, +0.35] |
| debate_seq | gpt-4o-mini | 3 | honest | yes | +0.01 | +0.10 [+0.03, +0.16] |
| debate_seq | gpt-4o-mini | 3 | liar | no | +0.00 | +0.00 |
| debate_seq | gpt-4o-mini | 4 | honest | yes | +0.04 | -0.14 [-0.21, -0.06] |
| debate_seq | gpt-4o-mini | 4 | liar | no | +0.00 | +0.00 |
| debate_seq | llama-3.1-8b | 1 | honest | yes | +0.06 | -0.04 [-0.12, +0.05] |
| debate_seq | llama-3.1-8b | 1 | liar | yes | +0.34 | +0.19 [+0.09, +0.31] |
| debate_seq | llama-3.1-8b | 2 | honest | yes | +0.05 | -0.16 [-0.28, -0.06] |
| debate_seq | llama-3.1-8b | 2 | liar | no | +0.00 | +0.00 |
| debate_seq | llama-3.1-8b | 3 | honest | yes | +0.16 | +0.09 [+0.02, +0.15] |
| debate_seq | llama-3.1-8b | 3 | liar | no | +0.00 | +0.00 |
| debate_seq | llama-3.1-8b | 4 | honest | yes | +0.12 | +0.15 [+0.07, +0.23] |
| debate_seq | llama-3.1-8b | 4 | liar | yes | +0.31 | +0.14 [+0.07, +0.21] |
| debate_sim | gemma-3-12b | 1 | honest | yes | +0.08 | +0.12 [-0.01, +0.26] |
| debate_sim | gemma-3-12b | 1 | liar | yes | +0.66 | +0.26 [+0.06, +0.46] |
| debate_sim | gemma-3-12b | 2 | honest | yes | +0.22 | -0.08 [-0.18, +0.02] |
| debate_sim | gemma-3-12b | 2 | liar | no | +0.00 | +0.00 |
| debate_sim | gemma-3-12b | 3 | honest | yes | +0.01 | +0.15 [+0.05, +0.26] |
| debate_sim | gemma-3-12b | 3 | liar | yes | +0.35 | +0.00 [-0.12, +0.13] |
| debate_sim | gemma-3-12b | 4 | honest | no | +0.00 | +0.00 |
| debate_sim | gemma-3-12b | 4 | liar | no | +0.00 | +0.00 |
| debate_sim | gpt-4o-mini | 1 | honest | yes | +0.04 | -0.13 [-0.24, -0.03] |
| debate_sim | gpt-4o-mini | 1 | liar | yes | +0.35 | +0.14 [+0.02, +0.26] |
| debate_sim | gpt-4o-mini | 2 | honest | yes | +0.04 | -0.02 [-0.06, +0.03] |
| debate_sim | gpt-4o-mini | 2 | liar | no | +0.00 | +0.00 |
| debate_sim | gpt-4o-mini | 3 | honest | yes | +0.06 | +0.11 [+0.06, +0.16] |
| debate_sim | gpt-4o-mini | 3 | liar | yes | +0.07 | +0.01 [-0.07, +0.08] |
| debate_sim | gpt-4o-mini | 4 | honest | yes | +0.03 | -0.01 [-0.07, +0.04] |
| debate_sim | gpt-4o-mini | 4 | liar | no | +0.00 | +0.00 |
| debate_sim | llama-3.1-8b | 1 | honest | yes | +0.15 | +0.06 [-0.02, +0.14] |
| debate_sim | llama-3.1-8b | 1 | liar | yes | +0.32 | +0.16 [+0.05, +0.28] |
| debate_sim | llama-3.1-8b | 2 | honest | yes | +0.26 | +0.03 [-0.05, +0.12] |
| debate_sim | llama-3.1-8b | 2 | liar | no | +0.00 | +0.00 |
| debate_sim | llama-3.1-8b | 3 | honest | no | +0.00 | +0.00 |
| debate_sim | llama-3.1-8b | 3 | liar | yes | +0.20 | -0.20 [-0.32, -0.08] |
| debate_sim | llama-3.1-8b | 4 | honest | yes | +0.10 | +0.02 [-0.04, +0.08] |
| debate_sim | llama-3.1-8b | 4 | liar | yes | +0.02 | +0.00 [-0.11, +0.11] |
