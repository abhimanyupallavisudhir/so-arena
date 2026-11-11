# Prover-estimator debate: incentives of both sides

Fixed payoff (`prover_estimator_v2`), gpt-5-mini as prover and estimator, 3 weak judges, 14400 episodes. 95% item-bootstrap intervals. Prover reward units (start term at most 0.125, total at most 1.875).

## Prover: reward arguing the truth minus lying, by estimator behaviour

| estimator | all judges | gpt-4o-mini | llama-3.1-8b | gemma-3-12b |
|---|---|---|---|---|
| sincere (as before) | -0.11 [-0.16, -0.07] | -0.13 [-0.19, -0.08] | -0.15 [-0.22, -0.08] | -0.06 [-0.12, +0.00] |
| predicts the judge | -0.12 [-0.16, -0.07] | -0.12 [-0.17, -0.07] | -0.15 [-0.20, -0.09] | -0.09 [-0.14, -0.03] |
| understates, mild | -0.07 [-0.11, -0.03] | -0.06 [-0.11, -0.01] | -0.14 [-0.20, -0.07] | -0.02 [-0.08, +0.04] |
| understates, strong | +0.05 [+0.01, +0.09] | +0.03 [-0.01, +0.08] | +0.04 [-0.02, +0.10] | +0.08 [+0.02, +0.14] |
| understates its first number only | -0.05 [-0.09, -0.01] | -0.05 [-0.10, +0.00] | -0.05 [-0.12, +0.01] | -0.04 [-0.10, +0.02] |
| overstates, mild | -0.14 [-0.18, -0.10] | -0.13 [-0.17, -0.09] | -0.18 [-0.24, -0.13] | -0.11 [-0.16, -0.06] |

## Estimator behaviours (judges pooled)

Estimator reward = minus the prover's. Accuracy = minus the squared error of its first probability (for the prover's answer) against the truth. Gain = reward minus the sincere estimator's on the same item and arm.

| estimator | reward | accuracy | gain over sincere | P(answer), true / false answers | prover's start / rounds / final terms |
|---|---|---|---|---|---|
| sincere (as before) | +0.031 [+0.005, +0.057] | -0.026 [-0.036, -0.018] | +0.000 [+0.000, +0.000] | 0.94 / 0.07 | +0.063 / +0.012 / -0.106 |
| predicts the judge | +0.029 [+0.004, +0.055] | -0.061 [-0.071, -0.052] | -0.001 [-0.032, +0.029] | 0.87 / 0.22 | +0.068 / +0.012 / -0.109 |
| understates, mild | +0.025 [+0.001, +0.047] | -0.042 [-0.052, -0.033] | -0.006 [-0.037, +0.023] | 0.80 / 0.07 | +0.054 / +0.006 / -0.086 |
| understates, strong | -0.085 [-0.107, -0.064] | -0.096 [-0.105, -0.088] | -0.116 [-0.143, -0.089] | 0.59 / 0.05 | +0.040 / +0.001 / +0.044 |
| understates its first number only | +0.013 [-0.012, +0.040] | -0.096 [-0.104, -0.088] | -0.017 [-0.049, +0.014] | 0.59 / 0.05 | +0.041 / +0.052 / -0.106 |
| overstates, mild | +0.038 [+0.015, +0.059] | -0.045 [-0.055, -0.037] | +0.007 [-0.023, +0.037] | 0.94 / 0.22 | +0.072 / +0.010 / -0.120 |

## Estimator EIC: does its reward rise with its accuracy?

Within each (item, arm, judge), across the 6 behaviours: covariance of reward and accuracy, slope of reward on accuracy, correlation; then averaged over items.

| prover's side | judge | covariance | slope | correlation | items with positive covariance |
|---|---|---|---|---|---|
| all | all | +0.0032 [+0.0022, +0.0041] | +0.77 [+0.32, +1.19] | +0.11 [+0.09, +0.13] | 66% |
| all | gemma-3-12b | +0.0030 [+0.0017, +0.0043] | +0.45 [-0.23, +1.05] | +0.12 [+0.08, +0.15] | 67% |
| all | gpt-4o-mini | +0.0029 [+0.0018, +0.0040] | +0.67 [+0.16, +1.16] | +0.14 [+0.11, +0.17] | 70% |
| all | llama-3.1-8b | +0.0035 [+0.0021, +0.0051] | +1.20 [+0.47, +1.90] | +0.08 [+0.05, +0.11] | 62% |
| prover honest | all | +0.0064 [+0.0049, +0.0080] | +0.96 [+0.78, +1.15] | +0.21 [+0.18, +0.24] | 70% |
| prover honest | gemma-3-12b | +0.0068 [+0.0049, +0.0088] | +0.99 [+0.73, +1.24] | +0.23 [+0.19, +0.27] | 70% |
| prover honest | gpt-4o-mini | +0.0057 [+0.0039, +0.0074] | +0.94 [+0.72, +1.15] | +0.24 [+0.20, +0.28] | 72% |
| prover honest | llama-3.1-8b | +0.0069 [+0.0048, +0.0091] | +0.96 [+0.67, +1.23] | +0.16 [+0.11, +0.20] | 63% |
| prover lying | all | -0.0001 [-0.0016, +0.0013] | +0.58 [-0.30, +1.42] | +0.02 [-0.02, +0.05] | 48% |
| prover lying | gemma-3-12b | -0.0008 [-0.0027, +0.0012] | -0.09 [-1.38, +1.05] | +0.00 [-0.04, +0.05] | 52% |
| prover lying | gpt-4o-mini | +0.0002 [-0.0013, +0.0016] | +0.40 [-0.61, +1.37] | +0.04 [-0.01, +0.09] | 54% |
| prover lying | llama-3.1-8b | +0.0002 [-0.0020, +0.0024] | +1.43 [-0.00, +2.86] | +0.01 [-0.04, +0.05] | 50% |
