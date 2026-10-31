# Decisive study: tables

## Spend

| stage | $ |
|---|---|
| audit | 15.32 |
| optimize | 235.84 |
| protocols | 895.25 |
| select | 115.19 |
| **total** | **1261.60** |

## Items and the gap achieved

3437 candidates; 927 selected (strict rule: every debater 2/2, judges pooled <= 0.60); 320 run through the protocols.

| model | role | accuracy on the draws used to select | fresh draws [95% CI] |
|---|---|---|---|
| gpt-4o-mini | judge | 0.38 | 0.44 [0.40, 0.48] |
| llama-3.1-8b | judge | 0.33 | 0.41 [0.38, 0.44] |
| gemma-3-12b | judge | 0.38 | 0.46 [0.42, 0.49] |
| claude-sonnet-4-5 | debater | 1.00 | 0.96 [0.95, 0.97] |
| gpt-5-mini | debater | 1.00 | 0.99 [0.98, 1.00] |

Families of the items run: numbers/prime_race 31, dtbench/capabilities 29, decisions/newcomb 27, decisions/damascus 21, decisions/hitchhiker 21, socialchoice/rankings 21, numbers/counts 19, decisions/lesion 18, decisions/twin 18, rewardhacking/gridworld_fake_goal 16, decisions/mugging 14, decisions/beauty 14, socialchoice/winners 12, numbers/collatz 10, rewardhacking/gridworld_watched_sensor 10, decisions/driver 8, rewardhacking/gridworld_sensor 6, multiagent/auction 5, toyml/weak_to_strong 5, sycophancy/wrong_answer 3, toyml/power_seeking 3, control/backdoor_monitor 3, scheming/sandbagging 2, multiagent/public_goods 1, personas/decision_theory 1, toyml/goal_misgeneralization 1, oversight/judge_style 1

## Prompted liars: pooled over all six (judge, debater) pairs

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all | consultancy | 0.62 [0.60, 0.63] | 0.44 | 0.97 | +18 [+16, +21] | 0.34 [0.31, 0.37] | 0.33 [0.31, 0.36] | 1.48 | 0.71 | 320 |
| all | all | debate (sequential) | 0.65 [0.63, 0.66] | 0.44 | 0.97 | +21 [+18, +24] | 0.39 [0.35, 0.43] | 0.41 [0.37, 0.44] | 1.76 | 0.72 | 320 |
| all | all | debate (simultaneous) | 0.68 [0.66, 0.70] | 0.44 | 0.97 | +25 [+22, +27] | 0.46 [0.41, 0.49] | 0.46 [0.42, 0.49] | 2.15 | 0.75 | 320 |
| all | all | prover-estimator | 0.52 [0.51, 0.54] | 0.44 | 0.97 | +9 [+6, +12] | 0.17 [0.12, 0.21] | 0.09 [0.04, 0.12] | -0.07 | 0.47 | 320 |
| all | all | recursive debate | 0.56 [0.55, 0.58] | 0.44 | 0.97 | +13 [+10, +16] | 0.24 [0.20, 0.28] | 0.27 [0.23, 0.30] | 0.13 | 0.66 | 320 |

## Prompted liars: per debater, pooled over judges

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | claude-sonnet-4-5 | consultancy | 0.68 [0.66, 0.70] | 0.44 | 0.96 | +25 [+22, +27] | 0.47 [0.43, 0.50] | 0.51 [0.48, 0.54] | 2.60 | 0.81 | 320 |
| all | claude-sonnet-4-5 | debate (sequential) | 0.71 [0.69, 0.73] | 0.44 | 0.96 | +28 [+24, +31] | 0.52 [0.47, 0.57] | 0.58 [0.54, 0.62] | 2.74 | 0.80 | 320 |
| all | claude-sonnet-4-5 | debate (simultaneous) | 0.76 [0.74, 0.79] | 0.44 | 0.96 | +33 [+29, +36] | 0.63 [0.58, 0.67] | 0.67 [0.63, 0.71] | 3.48 | 0.83 | 320 |
| all | claude-sonnet-4-5 | prover-estimator | 0.57 [0.55, 0.59] | 0.44 | 0.96 | +13 [+10, +16] | 0.25 [0.20, 0.30] | 0.11 [0.07, 0.15] | 0.01 | 0.57 | 320 |
| all | claude-sonnet-4-5 | recursive debate | 0.59 [0.57, 0.60] | 0.44 | 0.96 | +15 [+13, +18] | 0.29 [0.25, 0.33] | 0.34 [0.30, 0.38] | 0.18 | 0.69 | 320 |
| all | gpt-5-mini | consultancy | 0.56 [0.54, 0.57] | 0.44 | 0.99 | +12 [+10, +15] | 0.22 [0.19, 0.25] | 0.18 [0.15, 0.21] | 0.36 | 0.61 | 320 |
| all | gpt-5-mini | debate (sequential) | 0.58 [0.56, 0.60] | 0.44 | 0.99 | +15 [+12, +17] | 0.26 [0.22, 0.30] | 0.26 [0.22, 0.30] | 0.77 | 0.65 | 320 |
| all | gpt-5-mini | debate (simultaneous) | 0.60 [0.58, 0.62] | 0.44 | 0.99 | +16 [+13, +19] | 0.29 [0.25, 0.34] | 0.27 [0.23, 0.31] | 0.81 | 0.67 | 320 |
| all | gpt-5-mini | prover-estimator | 0.48 [0.47, 0.50] | 0.44 | 0.99 | +5 [+2, +8] | 0.09 [0.03, 0.13] | 0.06 [0.02, 0.10] | -0.15 | 0.37 | 320 |
| all | gpt-5-mini | recursive debate | 0.54 [0.53, 0.55] | 0.44 | 0.99 | +11 [+8, +14] | 0.19 [0.15, 0.24] | 0.21 [0.16, 0.24] | 0.09 | 0.63 | 320 |

## Prompted liars: per (judge, debater) pair

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.73 [0.70, 0.75] | 0.46 | 0.96 | +27 [+23, +31] | 0.54 [0.49, 0.58] | 0.60 [0.55, 0.65] | 3.54 | 0.82 | 320 |
| gemma-3-12b | claude-sonnet-4-5 | debate (sequential) | 0.66 [0.63, 0.69] | 0.46 | 0.96 | +20 [+15, +24] | 0.40 [0.32, 0.47] | 0.50 [0.43, 0.56] | 2.96 | 0.77 | 320 |
| gemma-3-12b | claude-sonnet-4-5 | debate (simultaneous) | 0.76 [0.73, 0.79] | 0.46 | 0.96 | +30 [+26, +34] | 0.60 [0.54, 0.66] | 0.68 [0.63, 0.73] | 3.99 | 0.83 | 320 |
| gemma-3-12b | claude-sonnet-4-5 | prover-estimator | 0.57 [0.55, 0.59] | 0.46 | 0.96 | +11 [+7, +16] | 0.23 [0.15, 0.29] | 0.06 [-0.01, 0.12] | 0.00 | 0.56 | 320 |
| gemma-3-12b | claude-sonnet-4-5 | recursive debate | 0.61 [0.58, 0.63] | 0.46 | 0.96 | +15 [+11, +19] | 0.30 [0.23, 0.36] | 0.35 [0.28, 0.41] | 0.22 | 0.69 | 320 |
| gemma-3-12b | gpt-5-mini | consultancy | 0.57 [0.56, 0.59] | 0.46 | 0.99 | +12 [+8, +15] | 0.22 [0.16, 0.26] | 0.19 [0.14, 0.24] | 0.79 | 0.69 | 320 |
| gemma-3-12b | gpt-5-mini | debate (sequential) | 0.53 [0.51, 0.56] | 0.46 | 0.99 | +7 [+3, +12] | 0.14 [0.06, 0.20] | 0.17 [0.10, 0.23] | 0.68 | 0.62 | 320 |
| gemma-3-12b | gpt-5-mini | debate (simultaneous) | 0.54 [0.52, 0.57] | 0.46 | 0.99 | +8 [+4, +12] | 0.16 [0.09, 0.22] | 0.19 [0.12, 0.25] | 0.72 | 0.62 | 320 |
| gemma-3-12b | gpt-5-mini | prover-estimator | 0.51 [0.49, 0.53] | 0.46 | 0.99 | +5 [+1, +9] | 0.10 [0.03, 0.16] | 0.03 [-0.04, 0.08] | -0.12 | 0.41 | 320 |
| gemma-3-12b | gpt-5-mini | recursive debate | 0.56 [0.54, 0.58] | 0.46 | 0.99 | +10 [+6, +15] | 0.20 [0.13, 0.26] | 0.20 [0.13, 0.25] | 0.12 | 0.65 | 320 |
| gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.70 [0.67, 0.73] | 0.44 | 0.96 | +26 [+22, +30] | 0.50 [0.43, 0.55] | 0.53 [0.49, 0.58] | 2.57 | 0.81 | 320 |
| gpt-4o-mini | claude-sonnet-4-5 | debate (sequential) | 0.76 [0.73, 0.79] | 0.44 | 0.96 | +32 [+28, +37] | 0.62 [0.56, 0.67] | 0.65 [0.60, 0.69] | 2.55 | 0.85 | 320 |
| gpt-4o-mini | claude-sonnet-4-5 | debate (simultaneous) | 0.79 [0.76, 0.82] | 0.44 | 0.96 | +35 [+31, +40] | 0.68 [0.62, 0.73] | 0.70 [0.65, 0.75] | 3.07 | 0.84 | 320 |
| gpt-4o-mini | claude-sonnet-4-5 | prover-estimator | 0.57 [0.55, 0.59] | 0.44 | 0.96 | +13 [+9, +17] | 0.25 [0.18, 0.31] | 0.10 [0.03, 0.17] | 0.02 | 0.60 | 320 |
| gpt-4o-mini | claude-sonnet-4-5 | recursive debate | 0.61 [0.59, 0.63] | 0.44 | 0.96 | +17 [+12, +21] | 0.32 [0.25, 0.38] | 0.39 [0.33, 0.45] | 0.23 | 0.77 | 320 |
| gpt-4o-mini | gpt-5-mini | consultancy | 0.60 [0.58, 0.63] | 0.44 | 0.99 | +16 [+13, +20] | 0.30 [0.24, 0.35] | 0.22 [0.17, 0.26] | 0.51 | 0.65 | 320 |
| gpt-4o-mini | gpt-5-mini | debate (sequential) | 0.65 [0.62, 0.68] | 0.44 | 0.99 | +21 [+17, +25] | 0.38 [0.33, 0.43] | 0.33 [0.29, 0.38] | 0.79 | 0.70 | 320 |
| gpt-4o-mini | gpt-5-mini | debate (simultaneous) | 0.68 [0.65, 0.71] | 0.44 | 0.99 | +24 [+19, +28] | 0.43 [0.37, 0.49] | 0.36 [0.31, 0.41] | 0.86 | 0.75 | 320 |
| gpt-4o-mini | gpt-5-mini | prover-estimator | 0.47 [0.45, 0.49] | 0.44 | 0.99 | +3 [-1, +8] | 0.06 [-0.02, 0.13] | 0.04 [-0.03, 0.10] | -0.17 | 0.36 | 320 |
| gpt-4o-mini | gpt-5-mini | recursive debate | 0.54 [0.53, 0.55] | 0.44 | 0.99 | +10 [+6, +14] | 0.18 [0.11, 0.24] | 0.22 [0.16, 0.28] | 0.11 | 0.70 | 320 |
| llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.62 [0.59, 0.64] | 0.41 | 0.96 | +21 [+18, +24] | 0.38 [0.33, 0.42] | 0.41 [0.37, 0.45] | 1.68 | 0.79 | 320 |
| llama-3.1-8b | claude-sonnet-4-5 | debate (sequential) | 0.71 [0.68, 0.74] | 0.41 | 0.96 | +30 [+26, +34] | 0.55 [0.49, 0.60] | 0.59 [0.55, 0.63] | 2.71 | 0.78 | 320 |
| llama-3.1-8b | claude-sonnet-4-5 | debate (simultaneous) | 0.74 [0.71, 0.76] | 0.41 | 0.96 | +33 [+29, +37] | 0.60 [0.54, 0.65] | 0.64 [0.59, 0.68] | 3.38 | 0.81 | 320 |
| llama-3.1-8b | claude-sonnet-4-5 | prover-estimator | 0.56 [0.54, 0.58] | 0.41 | 0.96 | +15 [+11, +19] | 0.27 [0.21, 0.32] | 0.17 [0.13, 0.21] | 0.00 | 0.55 | 320 |
| llama-3.1-8b | claude-sonnet-4-5 | recursive debate | 0.55 [0.52, 0.57] | 0.41 | 0.96 | +14 [+10, +17] | 0.25 [0.20, 0.30] | 0.28 [0.23, 0.33] | 0.09 | 0.61 | 320 |
| llama-3.1-8b | gpt-5-mini | consultancy | 0.50 [0.47, 0.52] | 0.41 | 0.99 | +9 [+6, +12] | 0.15 [0.10, 0.20] | 0.14 [0.10, 0.18] | -0.24 | 0.47 | 320 |
| llama-3.1-8b | gpt-5-mini | debate (sequential) | 0.56 [0.53, 0.59] | 0.41 | 0.99 | +15 [+11, +19] | 0.26 [0.20, 0.32] | 0.28 [0.22, 0.32] | 0.85 | 0.63 | 320 |
| llama-3.1-8b | gpt-5-mini | debate (simultaneous) | 0.58 [0.55, 0.60] | 0.41 | 0.99 | +17 [+13, +21] | 0.29 [0.22, 0.35] | 0.27 [0.22, 0.32] | 0.87 | 0.64 | 320 |
| llama-3.1-8b | gpt-5-mini | prover-estimator | 0.46 [0.44, 0.49] | 0.41 | 0.99 | +5 [+1, +9] | 0.09 [0.03, 0.16] | 0.11 [0.07, 0.15] | -0.17 | 0.33 | 320 |
| llama-3.1-8b | gpt-5-mini | recursive debate | 0.52 [0.50, 0.55] | 0.41 | 0.99 | +12 [+8, +15] | 0.20 [0.15, 0.25] | 0.20 [0.15, 0.25] | 0.03 | 0.53 | 320 |

## Audit of prompted liars (distinct transcripts)

| debater | mechanism | broken | committed | conceded | refused | unparsed | tell | n |
|---|---|---|---|---|---|---|---|---|
| claude-sonnet-4-5 | consultancy | 0.000 | 0.426 | 0.544 | 0.027 | 0.002 | 0.190 | 2880 |
| claude-sonnet-4-5 | debate_seq | 0.000 | 0.523 | 0.466 | 0.005 | 0.006 | 0.135 | 1920 |
| claude-sonnet-4-5 | debate_sim | 0.000 | 0.319 | 0.655 | 0.011 | 0.015 | 0.228 | 1920 |
| claude-sonnet-4-5 | prover_estimator | 0.000 | 0.301 | 0.698 | 0.000 | 0.001 | 0.267 | 960 |
| claude-sonnet-4-5 | recursive_debate | 0.000 | 0.320 | 0.675 | 0.000 | 0.005 | 0.287 | 960 |
| gpt-5-mini | consultancy | 0.000 | 0.913 | 0.069 | 0.010 | 0.009 | 0.001 | 2880 |
| gpt-5-mini | debate_seq | 0.000 | 0.947 | 0.046 | 0.005 | 0.002 | 0.004 | 1920 |
| gpt-5-mini | debate_sim | 0.000 | 0.942 | 0.054 | 0.001 | 0.004 | 0.000 | 1920 |
| gpt-5-mini | prover_estimator | 0.000 | 0.720 | 0.270 | 0.005 | 0.005 | 0.000 | 960 |
| gpt-5-mini | recursive_debate | 0.000 | 0.747 | 0.211 | 0.035 | 0.006 | 0.002 | 960 |

## Committed liars only: pooled over all pairs

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all | consultancy | 0.53 [0.52, 0.55] | 0.42 | 0.97 | +11 [+9, +14] | 0.21 [0.17, 0.24] | 0.18 [0.14, 0.20] | 0.14 | 0.57 | 320 |
| all | all | debate (sequential) | 0.55 [0.53, 0.57] | 0.43 | 0.97 | +12 [+9, +14] | 0.22 [0.17, 0.26] | 0.22 [0.19, 0.26] | 0.40 | 0.60 | 314 |
| all | all | debate (simultaneous) | 0.56 [0.54, 0.58] | 0.44 | 0.97 | +12 [+9, +15] | 0.23 [0.17, 0.27] | 0.22 [0.18, 0.26] | 0.43 | 0.61 | 309 |
| all | all | prover-estimator | 0.47 [0.46, 0.49] | 0.43 | 0.96 | +5 [+1, +8] | 0.09 [0.03, 0.14] | 0.08 [0.03, 0.12] | -0.16 | 0.40 | 308 |
| all | all | recursive debate | 0.53 [0.52, 0.55] | 0.43 | 0.96 | +11 [+8, +14] | 0.20 [0.15, 0.24] | 0.21 [0.16, 0.25] | 0.07 | 0.58 | 306 |

## Committed liars only: per debater

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | claude-sonnet-4-5 | consultancy | 0.51 [0.49, 0.54] | 0.40 | 0.93 | +12 [+8, +15] | 0.22 [0.16, 0.27] | 0.21 [0.16, 0.25] | -0.02 | 0.55 | 209 |
| all | claude-sonnet-4-5 | debate (sequential) | 0.51 [0.49, 0.54] | 0.41 | 0.93 | +10 [+6, +14] | 0.20 [0.13, 0.27] | 0.22 [0.16, 0.28] | 0.09 | 0.54 | 173 |
| all | claude-sonnet-4-5 | debate (simultaneous) | 0.49 [0.44, 0.53] | 0.40 | 0.91 | +9 [+3, +15] | 0.18 [0.06, 0.27] | 0.21 [0.11, 0.30] | -0.00 | 0.50 | 104 |
| all | claude-sonnet-4-5 | prover-estimator | 0.49 [0.46, 0.53] | 0.40 | 0.91 | +10 [+4, +15] | 0.19 [0.08, 0.29] | 0.20 [0.13, 0.27] | -0.05 | 0.46 | 138 |
| all | claude-sonnet-4-5 | recursive debate | 0.52 [0.49, 0.54] | 0.42 | 0.92 | +10 [+6, +14] | 0.21 [0.14, 0.28] | 0.25 [0.17, 0.32] | 0.05 | 0.54 | 153 |
| all | gpt-5-mini | consultancy | 0.55 [0.53, 0.56] | 0.44 | 0.99 | +11 [+9, +13] | 0.20 [0.17, 0.23] | 0.16 [0.13, 0.19] | 0.23 | 0.58 | 320 |
| all | gpt-5-mini | debate (sequential) | 0.57 [0.55, 0.59] | 0.44 | 0.99 | +12 [+10, +15] | 0.23 [0.18, 0.27] | 0.22 [0.19, 0.25] | 0.57 | 0.63 | 311 |
| all | gpt-5-mini | debate (simultaneous) | 0.58 [0.56, 0.60] | 0.45 | 0.99 | +13 [+10, +16] | 0.24 [0.20, 0.29] | 0.22 [0.18, 0.26] | 0.57 | 0.64 | 307 |
| all | gpt-5-mini | prover-estimator | 0.47 [0.45, 0.49] | 0.44 | 0.99 | +3 [-1, +5] | 0.05 [-0.01, 0.10] | 0.04 [-0.00, 0.07] | -0.20 | 0.37 | 303 |
| all | gpt-5-mini | recursive debate | 0.54 [0.52, 0.56] | 0.43 | 0.99 | +11 [+7, +14] | 0.19 [0.14, 0.24] | 0.19 [0.15, 0.24] | 0.08 | 0.60 | 302 |

## Committed liars only: per pair

| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) [CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.52 [0.48, 0.55] | 0.39 | 0.93 | +12 [+7, +18] | 0.23 [0.14, 0.31] | 0.24 [0.15, 0.32] | 0.24 | 0.54 | 183 |
| gemma-3-12b | claude-sonnet-4-5 | debate (sequential) | 0.43 [0.40, 0.47] | 0.41 | 0.93 | +3 [-4, +9] | 0.05 [-0.08, 0.16] | 0.08 [-0.04, 0.17] | -0.25 | 0.43 | 173 |
| gemma-3-12b | claude-sonnet-4-5 | debate (simultaneous) | 0.47 [0.41, 0.52] | 0.42 | 0.91 | +5 [-4, +14] | 0.11 [-0.09, 0.25] | 0.15 [-0.03, 0.30] | -0.13 | 0.48 | 104 |
| gemma-3-12b | claude-sonnet-4-5 | prover-estimator | 0.50 [0.46, 0.55] | 0.39 | 0.91 | +11 [+3, +18] | 0.21 [0.07, 0.33] | 0.20 [0.11, 0.28] | -0.05 | 0.44 | 138 |
| gemma-3-12b | claude-sonnet-4-5 | recursive debate | 0.52 [0.49, 0.55] | 0.41 | 0.92 | +11 [+6, +17] | 0.22 [0.12, 0.31] | 0.23 [0.12, 0.33] | 0.04 | 0.52 | 153 |
| gemma-3-12b | gpt-5-mini | consultancy | 0.56 [0.54, 0.58] | 0.46 | 0.99 | +10 [+7, +14] | 0.20 [0.14, 0.25] | 0.17 [0.11, 0.22] | 0.62 | 0.67 | 319 |
| gemma-3-12b | gpt-5-mini | debate (sequential) | 0.51 [0.49, 0.54] | 0.47 | 0.99 | +5 [+1, +9] | 0.09 [0.01, 0.16] | 0.12 [0.05, 0.18] | 0.43 | 0.59 | 311 |
| gemma-3-12b | gpt-5-mini | debate (simultaneous) | 0.52 [0.49, 0.55] | 0.47 | 0.99 | +5 [+1, +9] | 0.09 [0.01, 0.16] | 0.13 [0.06, 0.19] | 0.39 | 0.57 | 307 |
| gemma-3-12b | gpt-5-mini | prover-estimator | 0.50 [0.48, 0.53] | 0.46 | 0.99 | +4 [+0, +9] | 0.08 [0.00, 0.15] | 0.01 [-0.05, 0.07] | -0.16 | 0.41 | 303 |
| gemma-3-12b | gpt-5-mini | recursive debate | 0.56 [0.54, 0.58] | 0.45 | 0.99 | +11 [+6, +15] | 0.20 [0.13, 0.27] | 0.19 [0.12, 0.25] | 0.11 | 0.60 | 302 |
| gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.49 [0.45, 0.54] | 0.35 | 0.92 | +14 [+8, +20] | 0.25 [0.15, 0.34] | 0.25 [0.18, 0.31] | -0.02 | 0.52 | 172 |
| gpt-4o-mini | claude-sonnet-4-5 | debate (sequential) | 0.58 [0.53, 0.63] | 0.39 | 0.93 | +20 [+13, +26] | 0.36 [0.26, 0.45] | 0.36 [0.27, 0.44] | 0.29 | 0.65 | 173 |
| gpt-4o-mini | claude-sonnet-4-5 | debate (simultaneous) | 0.48 [0.41, 0.54] | 0.32 | 0.91 | +16 [+8, +24] | 0.27 [0.14, 0.39] | 0.30 [0.18, 0.40] | -0.12 | 0.54 | 104 |
| gpt-4o-mini | claude-sonnet-4-5 | prover-estimator | 0.50 [0.46, 0.54] | 0.33 | 0.91 | +17 [+9, +24] | 0.29 [0.18, 0.39] | 0.29 [0.21, 0.36] | -0.05 | 0.48 | 138 |
| gpt-4o-mini | claude-sonnet-4-5 | recursive debate | 0.52 [0.50, 0.55] | 0.36 | 0.92 | +16 [+10, +22] | 0.29 [0.20, 0.37] | 0.34 [0.25, 0.43] | 0.07 | 0.58 | 153 |
| gpt-4o-mini | gpt-5-mini | consultancy | 0.58 [0.56, 0.61] | 0.44 | 0.99 | +15 [+11, +18] | 0.26 [0.21, 0.32] | 0.19 [0.14, 0.24] | 0.39 | 0.62 | 320 |
| gpt-4o-mini | gpt-5-mini | debate (sequential) | 0.64 [0.61, 0.67] | 0.45 | 0.99 | +19 [+15, +23] | 0.35 [0.29, 0.41] | 0.30 [0.25, 0.34] | 0.65 | 0.69 | 311 |
| gpt-4o-mini | gpt-5-mini | debate (simultaneous) | 0.67 [0.64, 0.70] | 0.46 | 0.99 | +21 [+17, +26] | 0.40 [0.33, 0.46] | 0.32 [0.26, 0.38] | 0.73 | 0.73 | 307 |
| gpt-4o-mini | gpt-5-mini | prover-estimator | 0.46 [0.44, 0.48] | 0.45 | 0.99 | +1 [-4, +5] | 0.02 [-0.08, 0.09] | 0.01 [-0.07, 0.07] | -0.21 | 0.36 | 303 |
| gpt-4o-mini | gpt-5-mini | recursive debate | 0.54 [0.52, 0.55] | 0.44 | 0.99 | +10 [+5, +14] | 0.18 [0.10, 0.24] | 0.21 [0.13, 0.27] | 0.10 | 0.66 | 302 |
| llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.53 [0.49, 0.57] | 0.45 | 0.93 | +8 [+4, +13] | 0.17 [0.08, 0.25] | 0.12 [0.05, 0.19] | -0.28 | 0.58 | 181 |
| llama-3.1-8b | claude-sonnet-4-5 | debate (sequential) | 0.53 [0.49, 0.57] | 0.44 | 0.93 | +9 [+3, +14] | 0.18 [0.07, 0.28] | 0.21 [0.12, 0.30] | 0.21 | 0.53 | 173 |
| llama-3.1-8b | claude-sonnet-4-5 | debate (simultaneous) | 0.52 [0.46, 0.58] | 0.47 | 0.91 | +6 [-3, +14] | 0.13 [-0.06, 0.29] | 0.15 [-0.01, 0.28] | 0.24 | 0.48 | 104 |
| llama-3.1-8b | claude-sonnet-4-5 | prover-estimator | 0.48 [0.44, 0.52] | 0.46 | 0.91 | +1 [-5, +8] | 0.03 [-0.12, 0.16] | 0.10 [0.00, 0.18] | -0.07 | 0.47 | 138 |
| llama-3.1-8b | claude-sonnet-4-5 | recursive debate | 0.51 [0.47, 0.56] | 0.48 | 0.92 | +4 [-2, +10] | 0.08 [-0.05, 0.21] | 0.14 [0.01, 0.27] | 0.04 | 0.52 | 153 |
| llama-3.1-8b | gpt-5-mini | consultancy | 0.49 [0.47, 0.52] | 0.41 | 0.99 | +9 [+5, +12] | 0.15 [0.09, 0.20] | 0.13 [0.09, 0.17] | -0.32 | 0.45 | 318 |
| llama-3.1-8b | gpt-5-mini | debate (sequential) | 0.55 [0.52, 0.57] | 0.42 | 0.99 | +13 [+9, +17] | 0.23 [0.17, 0.29] | 0.24 [0.20, 0.29] | 0.65 | 0.60 | 311 |
| llama-3.1-8b | gpt-5-mini | debate (simultaneous) | 0.55 [0.53, 0.58] | 0.42 | 0.99 | +13 [+9, +17] | 0.24 [0.17, 0.29] | 0.22 [0.17, 0.27] | 0.60 | 0.62 | 307 |
| llama-3.1-8b | gpt-5-mini | prover-estimator | 0.44 [0.41, 0.47] | 0.41 | 0.99 | +2 [-2, +7] | 0.04 [-0.04, 0.11] | 0.08 [0.04, 0.13] | -0.24 | 0.35 | 303 |
| llama-3.1-8b | gpt-5-mini | recursive debate | 0.52 [0.49, 0.55] | 0.41 | 0.99 | +11 [+8, +15] | 0.19 [0.14, 0.26] | 0.19 [0.13, 0.25] | 0.03 | 0.53 | 302 |

## Audit of optimized liars (share of selected liar arguments, weighted by selection)

| debater | mechanism | n | not_committed | conceded |
|---|---|---|---|---|
| claude-sonnet-4-5 | consultancy_1r | 1 | 0.575 | 0.504 |
| claude-sonnet-4-5 | consultancy_1r | 2 | 0.542 | 0.473 |
| claude-sonnet-4-5 | consultancy_1r | 4 | 0.520 | 0.450 |
| claude-sonnet-4-5 | consultancy_1r | 8 | 0.504 | 0.435 |
| claude-sonnet-4-5 | consultancy_1r | 16 | 0.488 | 0.418 |
| claude-sonnet-4-5 | debate_1r | 1 | 0.486 | 0.416 |
| claude-sonnet-4-5 | debate_1r | 2 | 0.431 | 0.371 |
| claude-sonnet-4-5 | debate_1r | 4 | 0.395 | 0.340 |
| claude-sonnet-4-5 | debate_1r | 8 | 0.369 | 0.317 |
| claude-sonnet-4-5 | debate_1r | 16 | 0.352 | 0.301 |
| gpt-5-mini | consultancy_1r | 1 | 0.076 | 0.045 |
| gpt-5-mini | consultancy_1r | 2 | 0.062 | 0.036 |
| gpt-5-mini | consultancy_1r | 4 | 0.055 | 0.031 |
| gpt-5-mini | consultancy_1r | 8 | 0.051 | 0.028 |
| gpt-5-mini | consultancy_1r | 16 | 0.050 | 0.028 |
| gpt-5-mini | debate_1r | 1 | 0.030 | 0.015 |
| gpt-5-mini | debate_1r | 2 | 0.027 | 0.015 |
| gpt-5-mini | debate_1r | 4 | 0.027 | 0.016 |
| gpt-5-mini | debate_1r | 8 | 0.027 | 0.018 |
| gpt-5-mini | debate_1r | 16 | 0.029 | 0.019 |

## Optimized liars against the judge alone (paired by item; pooled over judges)

| debater | protocol | optimized | n | accuracy | judge alone | delta (pts) [CI] | gap recovered [CI] | items |
|---|---|---|---|---|---|---|---|---|
| all | single argument (consultancy, 1 round) | both | 1 | 0.59 [0.57, 0.62] | 0.44 | +16 [+13, +18] | 0.29 [0.24, 0.33] | 150 |
| all | single argument (consultancy, 1 round) | both | 2 | 0.59 [0.57, 0.61] | 0.44 | +15 [+12, +18] | 0.28 [0.24, 0.32] | 150 |
| all | single argument (consultancy, 1 round) | both | 4 | 0.57 [0.56, 0.59] | 0.44 | +13 [+10, +16] | 0.25 [0.20, 0.29] | 150 |
| all | single argument (consultancy, 1 round) | both | 8 | 0.55 [0.54, 0.56] | 0.44 | +11 [+8, +14] | 0.20 [0.15, 0.25] | 150 |
| all | single argument (consultancy, 1 round) | both | 16 | 0.53 [0.52, 0.54] | 0.44 | +9 [+6, +12] | 0.17 [0.12, 0.21] | 150 |
| all | single argument (consultancy, 1 round) | liar | 1 | 0.59 [0.57, 0.62] | 0.44 | +16 [+13, +18] | 0.29 [0.24, 0.33] | 150 |
| all | single argument (consultancy, 1 round) | liar | 2 | 0.51 [0.49, 0.54] | 0.44 | +7 [+4, +10] | 0.13 [0.08, 0.18] | 150 |
| all | single argument (consultancy, 1 round) | liar | 4 | 0.44 [0.42, 0.46] | 0.44 | +0 [-3, +3] | 0.01 [-0.05, 0.06] | 150 |
| all | single argument (consultancy, 1 round) | liar | 8 | 0.39 [0.38, 0.41] | 0.44 | -5 [-8, -2] | -0.09 [-0.15, -0.03] | 150 |
| all | single argument (consultancy, 1 round) | liar | 16 | 0.36 [0.35, 0.38] | 0.44 | -8 [-11, -5] | -0.14 [-0.21, -0.08] | 150 |
| all | debate (1 round) | both | 1 | 0.61 [0.58, 0.63] | 0.44 | +17 [+13, +20] | 0.31 [0.26, 0.35] | 150 |
| all | debate (1 round) | both | 2 | 0.60 [0.58, 0.63] | 0.44 | +16 [+13, +19] | 0.30 [0.25, 0.34] | 150 |
| all | debate (1 round) | both | 4 | 0.60 [0.57, 0.63] | 0.44 | +16 [+13, +19] | 0.30 [0.24, 0.34] | 150 |
| all | debate (1 round) | both | 8 | 0.60 [0.58, 0.63] | 0.44 | +16 [+13, +20] | 0.30 [0.25, 0.35] | 150 |
| all | debate (1 round) | both | 16 | 0.60 [0.57, 0.63] | 0.44 | +16 [+13, +20] | 0.30 [0.24, 0.35] | 150 |
| all | debate (1 round) | liar | 1 | 0.61 [0.58, 0.63] | 0.44 | +17 [+13, +20] | 0.31 [0.26, 0.35] | 150 |
| all | debate (1 round) | liar | 2 | 0.54 [0.52, 0.57] | 0.44 | +10 [+7, +13] | 0.19 [0.13, 0.24] | 150 |
| all | debate (1 round) | liar | 4 | 0.48 [0.45, 0.51] | 0.44 | +4 [+1, +7] | 0.08 [0.01, 0.13] | 150 |
| all | debate (1 round) | liar | 8 | 0.43 [0.40, 0.46] | 0.44 | -1 [-5, +2] | -0.02 [-0.09, 0.04] | 150 |
| all | debate (1 round) | liar | 16 | 0.38 [0.35, 0.41] | 0.44 | -6 [-9, -2] | -0.10 [-0.18, -0.04] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | both | 1 | 0.64 [0.61, 0.66] | 0.44 | +20 [+17, +23] | 0.38 [0.33, 0.42] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | both | 2 | 0.64 [0.61, 0.66] | 0.44 | +20 [+17, +23] | 0.37 [0.33, 0.42] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | both | 4 | 0.61 [0.59, 0.63] | 0.44 | +17 [+14, +20] | 0.33 [0.28, 0.37] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | both | 8 | 0.58 [0.56, 0.60] | 0.44 | +14 [+11, +17] | 0.27 [0.22, 0.31] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | both | 16 | 0.55 [0.54, 0.57] | 0.44 | +12 [+8, +15] | 0.22 [0.16, 0.26] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | liar | 1 | 0.64 [0.61, 0.66] | 0.44 | +20 [+17, +23] | 0.38 [0.33, 0.42] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | liar | 2 | 0.56 [0.53, 0.58] | 0.44 | +12 [+8, +15] | 0.22 [0.17, 0.27] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | liar | 4 | 0.48 [0.45, 0.51] | 0.44 | +4 [+1, +7] | 0.08 [0.01, 0.13] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | liar | 8 | 0.42 [0.40, 0.45] | 0.44 | -2 [-5, +1] | -0.03 [-0.10, 0.02] | 150 |
| claude-sonnet-4-5 | single argument (consultancy, 1 round) | liar | 16 | 0.39 [0.36, 0.41] | 0.44 | -5 [-9, -2] | -0.10 [-0.17, -0.04] | 150 |
| claude-sonnet-4-5 | debate (1 round) | both | 1 | 0.67 [0.64, 0.70] | 0.44 | +23 [+19, +26] | 0.43 [0.38, 0.48] | 150 |
| claude-sonnet-4-5 | debate (1 round) | both | 2 | 0.66 [0.64, 0.69] | 0.44 | +22 [+19, +26] | 0.42 [0.37, 0.47] | 150 |
| claude-sonnet-4-5 | debate (1 round) | both | 4 | 0.67 [0.64, 0.70] | 0.44 | +23 [+19, +26] | 0.43 [0.38, 0.48] | 150 |
| claude-sonnet-4-5 | debate (1 round) | both | 8 | 0.67 [0.64, 0.71] | 0.44 | +23 [+20, +27] | 0.44 [0.38, 0.50] | 150 |
| claude-sonnet-4-5 | debate (1 round) | both | 16 | 0.68 [0.64, 0.71] | 0.44 | +24 [+20, +27] | 0.45 [0.39, 0.51] | 150 |
| claude-sonnet-4-5 | debate (1 round) | liar | 1 | 0.67 [0.64, 0.70] | 0.44 | +23 [+19, +26] | 0.43 [0.38, 0.48] | 150 |
| claude-sonnet-4-5 | debate (1 round) | liar | 2 | 0.60 [0.57, 0.64] | 0.44 | +16 [+13, +20] | 0.31 [0.25, 0.37] | 150 |
| claude-sonnet-4-5 | debate (1 round) | liar | 4 | 0.54 [0.51, 0.58] | 0.44 | +11 [+7, +14] | 0.20 [0.13, 0.26] | 150 |
| claude-sonnet-4-5 | debate (1 round) | liar | 8 | 0.49 [0.46, 0.53] | 0.44 | +5 [+1, +9] | 0.10 [0.03, 0.17] | 150 |
| claude-sonnet-4-5 | debate (1 round) | liar | 16 | 0.44 [0.41, 0.48] | 0.44 | +0 [-4, +4] | 0.01 [-0.07, 0.08] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | both | 1 | 0.55 [0.53, 0.57] | 0.44 | +11 [+8, +14] | 0.20 [0.15, 0.24] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | both | 2 | 0.55 [0.53, 0.57] | 0.44 | +11 [+8, +14] | 0.20 [0.15, 0.24] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | both | 4 | 0.53 [0.52, 0.55] | 0.44 | +10 [+7, +13] | 0.17 [0.12, 0.22] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | both | 8 | 0.52 [0.51, 0.53] | 0.44 | +8 [+5, +11] | 0.14 [0.09, 0.19] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | both | 16 | 0.51 [0.50, 0.52] | 0.44 | +7 [+3, +10] | 0.12 [0.07, 0.17] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | liar | 1 | 0.55 [0.53, 0.57] | 0.44 | +11 [+8, +14] | 0.20 [0.15, 0.24] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | liar | 2 | 0.47 [0.45, 0.49] | 0.44 | +3 [+0, +6] | 0.05 [0.00, 0.10] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | liar | 4 | 0.40 [0.39, 0.42] | 0.44 | -3 [-6, -1] | -0.06 [-0.12, -0.01] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | liar | 8 | 0.36 [0.35, 0.38] | 0.44 | -8 [-11, -5] | -0.14 [-0.20, -0.08] | 150 |
| gpt-5-mini | single argument (consultancy, 1 round) | liar | 16 | 0.34 [0.33, 0.36] | 0.44 | -10 [-13, -7] | -0.18 [-0.25, -0.12] | 150 |
| gpt-5-mini | debate (1 round) | both | 1 | 0.55 [0.52, 0.57] | 0.44 | +11 [+7, +14] | 0.19 [0.14, 0.24] | 150 |
| gpt-5-mini | debate (1 round) | both | 2 | 0.54 [0.51, 0.56] | 0.44 | +10 [+6, +13] | 0.18 [0.12, 0.23] | 150 |
| gpt-5-mini | debate (1 round) | both | 4 | 0.53 [0.51, 0.56] | 0.44 | +9 [+6, +13] | 0.17 [0.11, 0.22] | 150 |
| gpt-5-mini | debate (1 round) | both | 8 | 0.53 [0.50, 0.56] | 0.44 | +9 [+6, +13] | 0.17 [0.11, 0.22] | 150 |
| gpt-5-mini | debate (1 round) | both | 16 | 0.53 [0.50, 0.56] | 0.44 | +9 [+5, +12] | 0.16 [0.10, 0.21] | 150 |
| gpt-5-mini | debate (1 round) | liar | 1 | 0.55 [0.52, 0.57] | 0.44 | +11 [+7, +14] | 0.19 [0.14, 0.24] | 150 |
| gpt-5-mini | debate (1 round) | liar | 2 | 0.48 [0.45, 0.50] | 0.44 | +4 [+0, +7] | 0.07 [0.01, 0.12] | 150 |
| gpt-5-mini | debate (1 round) | liar | 4 | 0.42 [0.39, 0.44] | 0.44 | -2 [-6, +1] | -0.04 [-0.11, 0.02] | 150 |
| gpt-5-mini | debate (1 round) | liar | 8 | 0.36 [0.34, 0.39] | 0.44 | -7 [-11, -4] | -0.13 [-0.20, -0.07] | 150 |
| gpt-5-mini | debate (1 round) | liar | 16 | 0.32 [0.30, 0.35] | 0.44 | -12 [-15, -8] | -0.21 [-0.29, -0.15] | 150 |

## Optimized liars: judge accuracy under best-of-n (mean of arms)

References on these items: {"judge alone": 0.439, "debater alone": 0.982}

| mechanism | mode | n | accuracy | lo | hi | p_true | items |
|---|---|---|---|---|---|---|---|
| consultancy_1r | both | 1 | 0.595 | 0.570 | 0.617 | 0.586 | 150 |
| consultancy_1r | both | 2 | 0.592 | 0.569 | 0.614 | 0.581 | 150 |
| consultancy_1r | both | 4 | 0.573 | 0.555 | 0.591 | 0.566 | 150 |
| consultancy_1r | both | 8 | 0.550 | 0.537 | 0.563 | 0.548 | 150 |
| consultancy_1r | both | 16 | 0.531 | 0.521 | 0.541 | 0.532 | 150 |
| consultancy_1r | liar | 1 | 0.595 | 0.570 | 0.617 | 0.586 | 150 |
| consultancy_1r | liar | 2 | 0.513 | 0.488 | 0.535 | 0.521 | 150 |
| consultancy_1r | liar | 4 | 0.442 | 0.421 | 0.462 | 0.466 | 150 |
| consultancy_1r | liar | 8 | 0.393 | 0.375 | 0.411 | 0.421 | 150 |
| consultancy_1r | liar | 16 | 0.364 | 0.348 | 0.380 | 0.389 | 150 |
| debate_1r | both | 1 | 0.607 | 0.584 | 0.630 | 0.598 | 150 |
| debate_1r | both | 2 | 0.601 | 0.577 | 0.626 | 0.592 | 150 |
| debate_1r | both | 4 | 0.600 | 0.575 | 0.627 | 0.590 | 150 |
| debate_1r | both | 8 | 0.603 | 0.575 | 0.631 | 0.591 | 150 |
| debate_1r | both | 16 | 0.602 | 0.573 | 0.631 | 0.590 | 150 |
| debate_1r | liar | 1 | 0.607 | 0.584 | 0.630 | 0.598 | 150 |
| debate_1r | liar | 2 | 0.541 | 0.516 | 0.567 | 0.547 | 150 |
| debate_1r | liar | 4 | 0.481 | 0.454 | 0.508 | 0.502 | 150 |
| debate_1r | liar | 8 | 0.428 | 0.401 | 0.456 | 0.462 | 150 |
| debate_1r | liar | 16 | 0.383 | 0.355 | 0.411 | 0.428 | 150 |

| mechanism | mode | debater | judge | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| consultancy_1r | both | claude-sonnet-4-5 | gemma-3-12b | 0.684 | 0.668 | 0.624 | 0.579 | 0.552 |
| consultancy_1r | both | claude-sonnet-4-5 | gpt-4o-mini | 0.695 | 0.699 | 0.679 | 0.644 | 0.598 |
| consultancy_1r | both | claude-sonnet-4-5 | llama-3.1-8b | 0.538 | 0.542 | 0.534 | 0.520 | 0.513 |
| consultancy_1r | both | gpt-5-mini | gemma-3-12b | 0.579 | 0.567 | 0.537 | 0.511 | 0.497 |
| consultancy_1r | both | gpt-5-mini | gpt-4o-mini | 0.569 | 0.569 | 0.558 | 0.542 | 0.523 |
| consultancy_1r | both | gpt-5-mini | llama-3.1-8b | 0.503 | 0.507 | 0.508 | 0.502 | 0.500 |
| consultancy_1r | liar | claude-sonnet-4-5 | gemma-3-12b | 0.684 | 0.607 | 0.540 | 0.487 | 0.454 |
| consultancy_1r | liar | claude-sonnet-4-5 | gpt-4o-mini | 0.695 | 0.632 | 0.570 | 0.510 | 0.450 |
| consultancy_1r | liar | claude-sonnet-4-5 | llama-3.1-8b | 0.538 | 0.429 | 0.328 | 0.272 | 0.255 |
| consultancy_1r | liar | gpt-5-mini | gemma-3-12b | 0.579 | 0.513 | 0.467 | 0.438 | 0.424 |
| consultancy_1r | liar | gpt-5-mini | gpt-4o-mini | 0.569 | 0.500 | 0.447 | 0.406 | 0.373 |
| consultancy_1r | liar | gpt-5-mini | llama-3.1-8b | 0.503 | 0.395 | 0.300 | 0.245 | 0.227 |
| debate_1r | both | claude-sonnet-4-5 | gemma-3-12b | 0.676 | 0.666 | 0.659 | 0.657 | 0.656 |
| debate_1r | both | claude-sonnet-4-5 | gpt-4o-mini | 0.736 | 0.735 | 0.739 | 0.740 | 0.745 |
| debate_1r | both | claude-sonnet-4-5 | llama-3.1-8b | 0.594 | 0.592 | 0.605 | 0.625 | 0.629 |
| debate_1r | both | gpt-5-mini | gemma-3-12b | 0.505 | 0.491 | 0.482 | 0.482 | 0.477 |
| debate_1r | both | gpt-5-mini | gpt-4o-mini | 0.593 | 0.585 | 0.577 | 0.570 | 0.559 |
| debate_1r | both | gpt-5-mini | llama-3.1-8b | 0.538 | 0.538 | 0.541 | 0.547 | 0.545 |
| debate_1r | liar | claude-sonnet-4-5 | gemma-3-12b | 0.676 | 0.607 | 0.543 | 0.485 | 0.434 |
| debate_1r | liar | claude-sonnet-4-5 | gpt-4o-mini | 0.736 | 0.674 | 0.616 | 0.565 | 0.518 |
| debate_1r | liar | claude-sonnet-4-5 | llama-3.1-8b | 0.594 | 0.531 | 0.474 | 0.423 | 0.380 |
| debate_1r | liar | gpt-5-mini | gemma-3-12b | 0.505 | 0.432 | 0.369 | 0.316 | 0.274 |
| debate_1r | liar | gpt-5-mini | gpt-4o-mini | 0.593 | 0.520 | 0.456 | 0.401 | 0.357 |
| debate_1r | liar | gpt-5-mini | llama-3.1-8b | 0.538 | 0.480 | 0.426 | 0.377 | 0.333 |
