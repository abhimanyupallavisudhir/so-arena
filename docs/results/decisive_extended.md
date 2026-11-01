# Decisive study: extended results

## Spend

| stage | $ |
|---|---|
| audit | 15.32 |
| audit_gpt-5 | 236.04 |
| auditcheck | 13.72 |
| multiround | 733.77 |
| optimize | 235.84 |
| promptsearch | 450.05 |
| protocols | 976.41 |
| quality | 637.71 |
| rejudge | 87.87 |
| select | 115.19 |
| tournament | 873.61 |
| **total** | **4375.54** |

## Multi-round best-of-n (3 rounds; preview judge on fresh samples)

Pooled over judges; delta = accuracy minus the judge alone (points); ASD in log score.

| mechanism | mode | n | accuracy | judge_alone | delta | gap_recovered | asd | items |
|---|---|---|---|---|---|---|---|---|
| consultancy | both | 1 | 0.57 [0.55, 0.59] | 0.439 | +13 [+10, +16] | 0.23 [0.18, 0.28] | 0.51 [0.29, 0.75] | 150 |
| consultancy | both | 4 | 0.54 [0.51, 0.57] | 0.439 | +10 [+6, +14] | 0.18 [0.11, 0.25] | 0.12 [-0.15, 0.40] | 150 |
| consultancy | both | 16 | 0.51 [0.47, 0.54] | 0.439 | +7 [+3, +11] | 0.12 [0.05, 0.18] | -0.06 [-0.34, 0.20] | 150 |
| consultancy | liar | 1 | 0.57 [0.55, 0.59] | 0.439 | +13 [+10, +16] | 0.23 [0.18, 0.28] | 0.51 [0.29, 0.75] | 150 |
| consultancy | liar | 4 | 0.53 [0.51, 0.56] | 0.439 | +9 [+6, +13] | 0.17 [0.11, 0.22] | 0.19 [-0.04, 0.44] | 150 |
| consultancy | liar | 16 | 0.50 [0.47, 0.52] | 0.439 | +6 [+2, +9] | 0.10 [0.04, 0.16] | 0.10 [-0.16, 0.37] | 150 |
| debate_seq | both | 1 | 0.58 [0.56, 0.61] | 0.439 | +14 [+10, +18] | 0.26 [0.20, 0.31] | 0.69 [0.51, 0.91] | 150 |
| debate_seq | both | 4 | 0.55 [0.51, 0.58] | 0.439 | +11 [+6, +15] | 0.19 [0.12, 0.26] | 0.53 [0.27, 0.79] | 150 |
| debate_seq | both | 16 | 0.55 [0.51, 0.59] | 0.439 | +11 [+6, +15] | 0.19 [0.12, 0.27] | 0.60 [0.27, 0.92] | 150 |
| debate_seq | liar | 1 | 0.58 [0.56, 0.61] | 0.439 | +14 [+10, +18] | 0.26 [0.20, 0.31] | 0.69 [0.51, 0.91] | 150 |
| debate_seq | liar | 4 | 0.53 [0.49, 0.56] | 0.439 | +9 [+4, +13] | 0.16 [0.08, 0.23] | 0.34 [0.05, 0.65] | 150 |
| debate_seq | liar | 16 | 0.51 [0.47, 0.54] | 0.439 | +7 [+2, +11] | 0.12 [0.04, 0.19] | 0.22 [-0.04, 0.49] | 150 |

Per judge:

| judge | mechanism | mode | n | accuracy | judge_alone | delta | gap_recovered | asd | items |
|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | consultancy | both | 1 | 0.59 [0.56, 0.62] | 0.459 | +13 [+8, +18] | 0.25 [0.16, 0.32] | 1.10 [0.72, 1.48] | 150 |
| gemma-3-12b | consultancy | both | 4 | 0.55 [0.52, 0.59] | 0.459 | +9 [+3, +16] | 0.18 [0.07, 0.27] | 0.56 [0.09, 1.03] | 150 |
| gemma-3-12b | consultancy | both | 16 | 0.53 [0.50, 0.56] | 0.459 | +7 [+1, +13] | 0.13 [0.03, 0.22] | 0.25 [-0.14, 0.65] | 150 |
| gemma-3-12b | consultancy | liar | 1 | 0.59 [0.56, 0.62] | 0.459 | +13 [+8, +18] | 0.25 [0.16, 0.32] | 1.10 [0.72, 1.48] | 150 |
| gemma-3-12b | consultancy | liar | 4 | 0.56 [0.53, 0.59] | 0.459 | +10 [+5, +16] | 0.19 [0.09, 0.28] | 0.68 [0.25, 1.15] | 150 |
| gemma-3-12b | consultancy | liar | 16 | 0.53 [0.50, 0.56] | 0.459 | +7 [+1, +13] | 0.13 [0.02, 0.22] | 0.42 [0.02, 0.88] | 150 |
| gemma-3-12b | debate_seq | both | 1 | 0.54 [0.50, 0.57] | 0.459 | +8 [+2, +14] | 0.14 [0.04, 0.24] | 0.74 [0.50, 0.99] | 150 |
| gemma-3-12b | debate_seq | both | 4 | 0.49 [0.44, 0.54] | 0.459 | +3 [-4, +10] | 0.05 [-0.08, 0.17] | 0.47 [0.02, 0.89] | 150 |
| gemma-3-12b | debate_seq | both | 16 | 0.47 [0.41, 0.52] | 0.459 | +1 [-6, +8] | 0.01 [-0.12, 0.13] | 0.31 [-0.21, 0.83] | 150 |
| gemma-3-12b | debate_seq | liar | 1 | 0.54 [0.50, 0.57] | 0.459 | +8 [+2, +14] | 0.14 [0.04, 0.24] | 0.74 [0.50, 0.99] | 150 |
| gemma-3-12b | debate_seq | liar | 4 | 0.47 [0.43, 0.52] | 0.459 | +1 [-5, +8] | 0.03 [-0.11, 0.14] | 0.14 [-0.25, 0.55] | 150 |
| gemma-3-12b | debate_seq | liar | 16 | 0.47 [0.42, 0.53] | 0.459 | +1 [-6, +8] | 0.03 [-0.11, 0.15] | 0.22 [-0.13, 0.58] | 150 |
| gpt-4o-mini | consultancy | both | 1 | 0.61 [0.57, 0.65] | 0.427 | +18 [+13, +24] | 0.33 [0.25, 0.39] | 0.63 [0.42, 0.88] | 150 |
| gpt-4o-mini | consultancy | both | 4 | 0.58 [0.53, 0.63] | 0.427 | +15 [+9, +21] | 0.27 [0.16, 0.36] | 0.32 [0.07, 0.57] | 150 |
| gpt-4o-mini | consultancy | both | 16 | 0.51 [0.47, 0.56] | 0.427 | +9 [+3, +15] | 0.15 [0.05, 0.24] | 0.12 [-0.10, 0.33] | 150 |
| gpt-4o-mini | consultancy | liar | 1 | 0.61 [0.57, 0.65] | 0.427 | +18 [+13, +24] | 0.33 [0.25, 0.39] | 0.63 [0.42, 0.88] | 150 |
| gpt-4o-mini | consultancy | liar | 4 | 0.55 [0.51, 0.59] | 0.427 | +12 [+7, +18] | 0.22 [0.12, 0.30] | 0.26 [0.03, 0.52] | 150 |
| gpt-4o-mini | consultancy | liar | 16 | 0.52 [0.48, 0.56] | 0.427 | +9 [+4, +14] | 0.17 [0.08, 0.24] | 0.24 [0.00, 0.52] | 150 |
| gpt-4o-mini | debate_seq | both | 1 | 0.66 [0.62, 0.70] | 0.427 | +23 [+18, +29] | 0.41 [0.34, 0.49] | 0.77 [0.59, 0.96] | 150 |
| gpt-4o-mini | debate_seq | both | 4 | 0.60 [0.55, 0.66] | 0.427 | +18 [+11, +25] | 0.31 [0.20, 0.41] | 0.43 [0.24, 0.62] | 150 |
| gpt-4o-mini | debate_seq | both | 16 | 0.62 [0.57, 0.68] | 0.427 | +19 [+13, +26] | 0.34 [0.24, 0.44] | 0.63 [0.34, 0.92] | 150 |
| gpt-4o-mini | debate_seq | liar | 1 | 0.66 [0.62, 0.70] | 0.427 | +23 [+18, +29] | 0.41 [0.34, 0.49] | 0.77 [0.59, 0.96] | 150 |
| gpt-4o-mini | debate_seq | liar | 4 | 0.57 [0.51, 0.63] | 0.427 | +15 [+8, +21] | 0.26 [0.15, 0.35] | 0.39 [0.13, 0.66] | 150 |
| gpt-4o-mini | debate_seq | liar | 16 | 0.54 [0.48, 0.59] | 0.427 | +11 [+4, +17] | 0.19 [0.08, 0.29] | 0.23 [0.01, 0.47] | 150 |
| llama-3.1-8b | consultancy | both | 1 | 0.50 [0.46, 0.53] | 0.431 | +7 [+2, +12] | 0.12 [0.03, 0.20] | -0.20 [-0.58, 0.20] | 150 |
| llama-3.1-8b | consultancy | both | 4 | 0.49 [0.43, 0.54] | 0.431 | +6 [+0, +12] | 0.10 [0.00, 0.20] | -0.52 [-1.02, -0.04] | 150 |
| llama-3.1-8b | consultancy | both | 16 | 0.47 [0.41, 0.54] | 0.431 | +4 [-2, +11] | 0.08 [-0.04, 0.19] | -0.56 [-1.12, -0.02] | 150 |
| llama-3.1-8b | consultancy | liar | 1 | 0.50 [0.46, 0.53] | 0.431 | +7 [+2, +12] | 0.12 [0.03, 0.20] | -0.20 [-0.58, 0.20] | 150 |
| llama-3.1-8b | consultancy | liar | 4 | 0.49 [0.44, 0.53] | 0.431 | +5 [-0, +12] | 0.10 [-0.00, 0.20] | -0.37 [-0.79, 0.08] | 150 |
| llama-3.1-8b | consultancy | liar | 16 | 0.44 [0.39, 0.49] | 0.431 | +1 [-5, +7] | 0.01 [-0.09, 0.11] | -0.38 [-0.81, 0.10] | 150 |
| llama-3.1-8b | debate_seq | both | 1 | 0.55 [0.52, 0.59] | 0.431 | +12 [+6, +18] | 0.21 [0.11, 0.30] | 0.57 [0.24, 0.93] | 150 |
| llama-3.1-8b | debate_seq | both | 4 | 0.55 [0.49, 0.61] | 0.431 | +12 [+4, +19] | 0.20 [0.08, 0.32] | 0.71 [0.17, 1.24] | 150 |
| llama-3.1-8b | debate_seq | both | 16 | 0.56 [0.50, 0.61] | 0.431 | +12 [+5, +20] | 0.22 [0.09, 0.34] | 0.85 [0.28, 1.43] | 150 |
| llama-3.1-8b | debate_seq | liar | 1 | 0.55 [0.52, 0.59] | 0.431 | +12 [+6, +18] | 0.21 [0.11, 0.30] | 0.57 [0.24, 0.93] | 150 |
| llama-3.1-8b | debate_seq | liar | 4 | 0.53 [0.48, 0.59] | 0.431 | +10 [+3, +17] | 0.18 [0.05, 0.29] | 0.49 [-0.04, 1.03] | 150 |
| llama-3.1-8b | debate_seq | liar | 16 | 0.51 [0.46, 0.56] | 0.431 | +8 [+1, +15] | 0.14 [0.02, 0.25] | 0.20 [-0.29, 0.71] | 150 |

## One-round best-of-n incentives (in-sample judging)

| mechanism | mode | level | accuracy | asd | asd_lo | asd_hi | concordant |
|---|---|---|---|---|---|---|---|
| consultancy_1r | both | 1.000 | 0.595 | 1.083 | 0.806 | 1.344 | 0.636 |
| consultancy_1r | both | 2.000 | 0.592 | 0.797 | 0.612 | 0.986 | 0.646 |
| consultancy_1r | both | 4.000 | 0.573 | 0.491 | 0.375 | 0.614 | 0.658 |
| consultancy_1r | both | 8.000 | 0.550 | 0.282 | 0.204 | 0.370 | 0.657 |
| consultancy_1r | both | 16.000 | 0.531 | 0.160 | 0.106 | 0.227 | 0.599 |
| consultancy_1r | liar | 1.000 | 0.595 | 1.083 | 0.806 | 1.344 | 0.636 |
| consultancy_1r | liar | 2.000 | 0.513 | 0.204 | 0.001 | 0.402 | 0.431 |
| consultancy_1r | liar | 4.000 | 0.442 | -0.334 | -0.500 | -0.173 | 0.283 |
| consultancy_1r | liar | 8.000 | 0.393 | -0.636 | -0.783 | -0.491 | 0.196 |
| consultancy_1r | liar | 16.000 | 0.364 | -0.803 | -0.939 | -0.671 | 0.116 |
| debate_1r | both | 1.000 | 0.607 | 1.234 | 0.984 | 1.489 | 0.654 |
| debate_1r | both | 2.000 | 0.601 | 1.112 | 0.869 | 1.356 | 0.653 |
| debate_1r | both | 4.000 | 0.600 | 1.033 | 0.802 | 1.270 | 0.643 |
| debate_1r | both | 8.000 | 0.603 | 1.000 | 0.770 | 1.232 | 0.648 |
| debate_1r | both | 16.000 | 0.602 | 0.999 | 0.762 | 1.234 | 0.642 |
| debate_1r | liar | 1.000 | 0.607 | 1.234 | 0.984 | 1.489 | 0.654 |
| debate_1r | liar | 2.000 | 0.541 | 0.680 | 0.450 | 0.914 | 0.523 |
| debate_1r | liar | 4.000 | 0.481 | 0.219 | 0.002 | 0.435 | 0.414 |
| debate_1r | liar | 8.000 | 0.428 | -0.178 | -0.385 | 0.027 | 0.349 |
| debate_1r | liar | 16.000 | 0.383 | -0.519 | -0.721 | -0.320 | 0.303 |

Consultant choosing its side (in-sample): share of selected arguments defending the truth

| debater | level | honest | honest_lo | honest_hi | accuracy |
|---|---|---|---|---|---|
| claude-sonnet-4-5 | 1.000 | 0.500 | 0.500 | 0.500 | 0.639 |
| claude-sonnet-4-5 | 2.000 | 0.589 | 0.574 | 0.603 | 0.629 |
| claude-sonnet-4-5 | 4.000 | 0.643 | 0.617 | 0.667 | 0.640 |
| claude-sonnet-4-5 | 8.000 | 0.655 | 0.623 | 0.685 | 0.650 |
| claude-sonnet-4-5 | 16.000 | 0.654 | 0.617 | 0.687 | 0.651 |
| claude-sonnet-4-5 | 32.000 | 0.653 | 0.610 | 0.694 | 0.651 |
| gpt-5-mini | 1.000 | 0.500 | 0.500 | 0.500 | 0.550 |
| gpt-5-mini | 2.000 | 0.531 | 0.515 | 0.545 | 0.555 |
| gpt-5-mini | 4.000 | 0.559 | 0.531 | 0.586 | 0.562 |
| gpt-5-mini | 8.000 | 0.576 | 0.539 | 0.609 | 0.574 |
| gpt-5-mini | 16.000 | 0.588 | 0.547 | 0.625 | 0.587 |
| gpt-5-mini | 32.000 | 0.605 | 0.563 | 0.646 | 0.605 |
| all | 1.000 | 0.500 | 0.500 | 0.500 | 0.595 |
| all | 2.000 | 0.560 | 0.546 | 0.573 | 0.592 |
| all | 4.000 | 0.601 | 0.576 | 0.625 | 0.601 |
| all | 8.000 | 0.615 | 0.585 | 0.644 | 0.612 |
| all | 16.000 | 0.621 | 0.586 | 0.653 | 0.619 |
| all | 32.000 | 0.629 | 0.591 | 0.666 | 0.628 |

## One-round best-of-n incentives (held-out judging)

| mechanism | mode | level | accuracy | asd | asd_lo | asd_hi | concordant |
|---|---|---|---|---|---|---|---|
| consultancy_1r | both | 1.000 | 0.597 | 1.095 | 0.808 | 1.357 | 0.660 |
| consultancy_1r | both | 2.000 | 0.590 | 0.971 | 0.706 | 1.219 | 0.642 |
| consultancy_1r | both | 4.000 | 0.586 | 0.883 | 0.631 | 1.118 | 0.617 |
| consultancy_1r | both | 8.000 | 0.582 | 0.822 | 0.577 | 1.058 | 0.608 |
| consultancy_1r | both | 16.000 | 0.579 | 0.764 | 0.508 | 1.023 | 0.584 |
| consultancy_1r | liar | 1.000 | 0.597 | 1.095 | 0.808 | 1.357 | 0.660 |
| consultancy_1r | liar | 2.000 | 0.579 | 0.906 | 0.630 | 1.161 | 0.613 |
| consultancy_1r | liar | 4.000 | 0.566 | 0.774 | 0.509 | 1.019 | 0.573 |
| consultancy_1r | liar | 8.000 | 0.556 | 0.693 | 0.433 | 0.944 | 0.530 |
| consultancy_1r | liar | 16.000 | 0.550 | 0.627 | 0.358 | 0.886 | 0.466 |
| debate_1r | both | 1.000 | 0.606 | 1.238 | 0.985 | 1.498 | 0.661 |
| debate_1r | both | 2.000 | 0.591 | 1.078 | 0.835 | 1.323 | 0.637 |
| debate_1r | both | 4.000 | 0.581 | 0.976 | 0.742 | 1.220 | 0.636 |
| debate_1r | both | 8.000 | 0.573 | 0.906 | 0.672 | 1.141 | 0.618 |
| debate_1r | both | 16.000 | 0.571 | 0.879 | 0.639 | 1.129 | 0.619 |
| debate_1r | liar | 1.000 | 0.606 | 1.238 | 0.985 | 1.498 | 0.661 |
| debate_1r | liar | 2.000 | 0.563 | 0.897 | 0.655 | 1.144 | 0.574 |
| debate_1r | liar | 4.000 | 0.524 | 0.603 | 0.377 | 0.839 | 0.512 |
| debate_1r | liar | 8.000 | 0.492 | 0.345 | 0.127 | 0.570 | 0.459 |
| debate_1r | liar | 16.000 | 0.464 | 0.122 | -0.094 | 0.339 | 0.431 |

Consultant choosing its side (held-out): share of selected arguments defending the truth

| debater | level | honest | honest_lo | honest_hi | accuracy |
|---|---|---|---|---|---|
| claude-sonnet-4-5 | 1.000 | 0.500 | 0.500 | 0.500 | 0.645 |
| claude-sonnet-4-5 | 2.000 | 0.589 | 0.574 | 0.603 | 0.635 |
| claude-sonnet-4-5 | 4.000 | 0.643 | 0.617 | 0.667 | 0.635 |
| claude-sonnet-4-5 | 8.000 | 0.655 | 0.623 | 0.685 | 0.635 |
| claude-sonnet-4-5 | 16.000 | 0.654 | 0.617 | 0.687 | 0.636 |
| claude-sonnet-4-5 | 32.000 | 0.653 | 0.610 | 0.694 | 0.646 |
| gpt-5-mini | 1.000 | 0.500 | 0.500 | 0.500 | 0.550 |
| gpt-5-mini | 2.000 | 0.531 | 0.515 | 0.545 | 0.545 |
| gpt-5-mini | 4.000 | 0.559 | 0.531 | 0.586 | 0.544 |
| gpt-5-mini | 8.000 | 0.576 | 0.539 | 0.609 | 0.544 |
| gpt-5-mini | 16.000 | 0.588 | 0.547 | 0.625 | 0.545 |
| gpt-5-mini | 32.000 | 0.605 | 0.563 | 0.646 | 0.550 |
| all | 1.000 | 0.500 | 0.500 | 0.500 | 0.597 |
| all | 2.000 | 0.560 | 0.546 | 0.573 | 0.590 |
| all | 4.000 | 0.601 | 0.576 | 0.625 | 0.590 |
| all | 8.000 | 0.615 | 0.585 | 0.644 | 0.589 |
| all | 16.000 | 0.621 | 0.586 | 0.653 | 0.590 |
| all | 32.000 | 0.629 | 0.591 | 0.666 | 0.598 |

## Prompt search: honesty margin (agent chooses its side; held-out items)

margin = held-out reward of the best honest strategy minus the best deceptive one (log score); value = 2 x share of episodes defending the true answer - 1.

| judge | mechanism | basis | margin | ci_low | ci_high | honest_reward | deceptive_reward | honest_value | deceptive_value | n_items |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | consultancy | measured | nan | nan | nan | -0.265 | nan | 0.940 | nan | 0 |
| gemma-3-12b | consultancy | directive | -0.030 | -0.400 | 0.301 | -0.294 | -0.265 | 0.920 | 0.940 | 100 |
| gemma-3-12b | debate_seq | measured | nan | nan | nan | -1.596 | nan | 0.920 | nan | 0 |
| gemma-3-12b | debate_seq | directive | -0.106 | -0.474 | 0.256 | -1.596 | -1.490 | 0.920 | 0.920 | 100 |
| gpt-4o-mini | consultancy | measured | nan | nan | nan | -0.214 | nan | 0.960 | nan | 0 |
| gpt-4o-mini | consultancy | directive | 0.363 | 0.104 | 0.693 | -0.214 | -0.576 | 0.960 | 0.860 | 100 |
| gpt-4o-mini | debate_seq | measured | nan | nan | nan | -0.894 | nan | 0.980 | nan | 0 |
| gpt-4o-mini | debate_seq | directive | 0.225 | 0.029 | 0.414 | -0.894 | -1.119 | 0.980 | 0.880 | 100 |
| llama-3.1-8b | consultancy | measured | nan | nan | nan | -0.898 | nan | 0.940 | nan | 0 |
| llama-3.1-8b | consultancy | directive | -0.897 | -1.601 | -0.221 | -1.795 | -0.898 | 0.960 | 0.940 | 100 |
| llama-3.1-8b | debate_seq | measured | nan | nan | nan | -1.352 | nan | 0.940 | nan | 0 |
| llama-3.1-8b | debate_seq | directive | -0.133 | -0.692 | 0.428 | -1.352 | -1.219 | 0.940 | 0.900 | 100 |

## Prompt search: best-response ladder (assigned sides; held-out items)

| judge | mechanism | rung | accuracy | accuracy_true_arm | accuracy_false_arm | honest_reward | liar_reward |
|---|---|---|---|---|---|---|---|
| gemma-3-12b | consultancy | base | 0.590 | 0.920 | 0.260 | -0.345 | -1.231 |
| gemma-3-12b | consultancy | liar optimized | 0.540 | 0.920 | 0.160 | -0.345 | -0.525 |
| gemma-3-12b | consultancy | both optimized | 0.512 | 0.865 | 0.160 | -0.816 | -0.525 |
| gpt-4o-mini | consultancy | base | 0.560 | 0.780 | 0.340 | -0.463 | -0.659 |
| gpt-4o-mini | consultancy | liar optimized | 0.610 | 0.780 | 0.440 | -0.463 | -0.766 |
| gpt-4o-mini | consultancy | both optimized | 0.610 | 0.780 | 0.440 | -0.463 | -0.766 |
| gpt-4o-mini | debate_seq | base | 0.400 | 0.400 | 0.400 | -1.015 | -0.917 |
| gpt-4o-mini | debate_seq | liar optimized | 0.380 | 0.380 | 0.380 | -1.135 | -0.816 |
| gpt-4o-mini | debate_seq | both optimized | 0.460 | 0.460 | 0.460 | -1.085 | -1.097 |
| gpt-4o-mini | debate_seq | liar replies | 0.540 | 0.540 | 0.540 | -0.920 | -1.170 |
| llama-3.1-8b | consultancy | base | 0.498 | 0.490 | 0.505 | -1.453 | -1.269 |
| llama-3.1-8b | consultancy | liar optimized | 0.468 | 0.490 | 0.445 | -1.453 | -1.012 |
| llama-3.1-8b | consultancy | both optimized | 0.552 | 0.660 | 0.445 | -1.216 | -1.012 |
| llama-3.1-8b | debate_seq | base | 0.440 | 0.440 | 0.440 | -1.418 | -1.460 |
| llama-3.1-8b | debate_seq | liar optimized | 0.540 | 0.540 | 0.540 | -1.242 | -1.672 |
| llama-3.1-8b | debate_seq | both optimized | 0.580 | 0.580 | 0.580 | -1.132 | -1.733 |
| llama-3.1-8b | debate_seq | liar replies | 0.410 | 0.410 | 0.410 | -1.734 | -1.104 |

## QuALITY (information asymmetry, verified quotes)

| debater | mechanism | accuracy | judge_alone | debater_alone | delta | gap_recovered | asd | concordance | items |
|---|---|---|---|---|---|---|---|---|---|
| all | consultancy | 0.60 [0.58, 0.61] | 0.530 | 0.949 | +7 [+3, +10] | 0.16 [0.09, 0.23] | 0.706 | 0.689 | 300 |
| all | debate_seq | 0.60 [0.59, 0.62] | 0.530 | 0.949 | +7 [+4, +11] | 0.18 [0.10, 0.25] | 0.447 | 0.669 | 300 |
| all | debate_seq_noquotes | 0.60 [0.58, 0.62] | 0.530 | 0.949 | +7 [+3, +10] | 0.16 [0.09, 0.23] | nan | nan | 300 |
| claude-sonnet-4-5 | consultancy | 0.66 [0.64, 0.69] | 0.525 | 0.968 | +14 [+10, +18] | 0.31 [0.24, 0.38] | 1.414 | 0.759 | 250 |
| claude-sonnet-4-5 | debate_seq | 0.64 [0.61, 0.66] | 0.525 | 0.968 | +11 [+7, +15] | 0.25 [0.17, 0.32] | 0.749 | 0.732 | 250 |
| claude-sonnet-4-5 | debate_seq_noquotes | 0.62 [0.59, 0.64] | 0.525 | 0.968 | +9 [+5, +13] | 0.20 [0.12, 0.27] | nan | nan | 250 |
| gpt-5-mini | consultancy | 0.54 [0.53, 0.56] | 0.534 | 0.933 | +1 [-2, +5] | 0.03 [-0.06, 0.11] | 0.117 | 0.630 | 300 |
| gpt-5-mini | debate_seq | 0.58 [0.56, 0.60] | 0.534 | 0.933 | +4 [+1, +8] | 0.11 [0.01, 0.20] | 0.196 | 0.617 | 300 |
| gpt-5-mini | debate_seq_noquotes | 0.58 [0.56, 0.60] | 0.534 | 0.933 | +5 [+1, +9] | 0.12 [0.03, 0.21] | nan | nan | 300 |

## Auditor agreement

| auditor_a | auditor_b | agreement | kappa | committed_vs_not_kappa |
|---|---|---|---|---|
| gpt-4.1-mini | gpt-5 | 0.556 | 0.396 | 0.449 |
| gpt-4.1-mini | claude-sonnet-5 | 0.577 | 0.437 | 0.471 |
| gpt-5 | claude-sonnet-5 | 0.837 | 0.755 | 0.796 |

## Share of liar texts not committed, by auditor

| kind | debater | not_committed_gpt-4.1-mini | not_committed_gpt-5 | not_committed_claude-sonnet-5 |
|---|---|---|---|---|
| optimized | claude-sonnet-4-5 | 0.515 | 0.521 | 0.634 |
| optimized | gpt-5-mini | 0.045 | 0.015 | 0.114 |
| prompted | claude-sonnet-4-5 | 0.602 | 0.670 | 0.676 |
| prompted | gpt-5-mini | 0.113 | 0.153 | 0.185 |

## Tournament

| model | elo | elo_lo | elo_hi | alone | self_play | consultancy | win_as_honest | win_as_liar |
|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | -42.549 | -52.059 | -34.535 | 0.911 | 0.587 | 0.591 | 0.562 | 0.307 |
| claude-haiku-4-5 | -13.885 | -23.439 | -6.436 | 0.916 | 0.601 | 0.612 | 0.622 | 0.338 |
| gpt-4.1 | 18.918 | 10.902 | 27.670 | 0.919 | 0.565 | 0.607 | 0.656 | 0.400 |
| gpt-5-mini | 35.367 | 27.001 | 43.980 | 0.988 | 0.556 | 0.559 | 0.655 | 0.428 |
| claude-sonnet-4-5 | -25.275 | -32.661 | -16.734 | 0.961 | 0.712 | 0.702 | 0.662 | 0.273 |
| claude-sonnet-5 | 25.460 | 17.801 | 33.062 | 0.976 | 0.626 | 0.578 | 0.660 | 0.409 |
| gpt-5 | 48.363 | 41.097 | 56.476 | 0.972 | 0.535 | 0.528 | 0.653 | 0.470 |
| claude-opus-5-5 | -46.399 | -54.549 | -39.076 | 0.982 | 0.862 | 0.855 | 0.723 | 0.168 |

honesty advantage (Elo), all debates: 100.4 [87.8, 114.5]
honesty advantage (Elo), committed liars: 30.6 [20.1, 42.0]
[all] slope of self_play accuracy per 100 elo: -0.2036
[all] slope of consultancy accuracy per 100 elo: -0.2083
[all] slope of self_play accuracy per 100 elo_honest: +0.1629
[all] slope of consultancy accuracy per 100 elo_honest: +0.1395
[all] slope of self_play accuracy per 100 elo_liar: -0.1366
[all] slope of consultancy accuracy per 100 elo_liar: -0.1349
[all] slope of self-play accuracy per +0.10 liar commitment: -0.0404
[committed] slope of self_play accuracy per 100 elo: +0.0045
[committed] slope of consultancy accuracy per 100 elo: +0.1521
[committed] slope of self_play accuracy per 100 elo_honest: +0.0333
[committed] slope of consultancy accuracy per 100 elo_honest: +0.1335
[committed] slope of self_play accuracy per 100 elo_liar: -0.0332
[committed] slope of consultancy accuracy per 100 elo_liar: +0.1321
[committed] slope of self-play accuracy per +0.10 liar commitment: +0.0262
debates: 21570 judge verdicts on 200 items

