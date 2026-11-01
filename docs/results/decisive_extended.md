# Decisive study: extended results

## Spend

| stage | $ |
|---|---|
| audit | 15.32 |
| audit_gpt-5 | 308.77 |
| auditcheck | 13.72 |
| judges | 318.84 |
| multiround | 733.77 |
| optimize | 235.84 |
| promptsearch | 491.26 |
| protocols | 1157.20 |
| quality | 637.71 |
| rejudge | 87.87 |
| select | 115.19 |
| tournament | 873.61 |
| **total** | **4989.11** |

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

Liars under per-speech best-of-n (gpt-5 audit); accuracy on episodes with a committed liar (consultancy: its false arm):

| mechanism | mode | n | liar_committed | liar_conceded | episodes | false_arm_accuracy_committed | judge_alone_same_items |
|---|---|---|---|---|---|---|---|
| consultancy | both | 4 | 0.891 | 0.082 | 450 | 0.283 | 0.442 |
| consultancy | both | 16 | 0.889 | 0.087 | 450 | 0.209 | 0.445 |
| consultancy | liar | 4 | 0.902 | 0.071 | 450 | 0.299 | 0.445 |
| consultancy | liar | 16 | 0.896 | 0.087 | 450 | 0.227 | 0.447 |
| debate_seq | both | 4 | 0.938 | 0.053 | 900 | 0.528 | 0.444 |
| debate_seq | both | 16 | 0.947 | 0.048 | 900 | 0.536 | 0.443 |
| debate_seq | liar | 4 | 0.931 | 0.062 | 900 | 0.514 | 0.451 |
| debate_seq | liar | 16 | 0.949 | 0.047 | 900 | 0.496 | 0.445 |

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
| gemma-3-12b | debate_seq | base | 0.300 | 0.300 | 0.300 | -1.879 | -1.186 |
| gemma-3-12b | debate_seq | liar optimized | 0.350 | 0.350 | 0.350 | -1.390 | -1.084 |
| gemma-3-12b | debate_seq | both optimized | 0.350 | 0.350 | 0.350 | -1.390 | -1.084 |
| gemma-3-12b | debate_seq | liar replies | 0.350 | 0.350 | 0.350 | -1.390 | -1.084 |
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

QuALITY liars committed (gpt-5 audit):

| debater | mechanism | committed |
|---|---|---|
| claude-sonnet-4-5 | consultancy | 0.496 |
| claude-sonnet-4-5 | debate_seq | 0.781 |
| claude-sonnet-4-5 | debate_seq_noquotes | 0.786 |
| gpt-5-mini | consultancy | 0.958 |
| gpt-5-mini | debate_seq | 0.987 |
| gpt-5-mini | debate_seq_noquotes | 0.988 |

### QuALITY, committed liars only

| debater | mechanism | accuracy | judge_alone | debater_alone | delta | gap_recovered | asd | concordance | items |
|---|---|---|---|---|---|---|---|---|---|
| all | consultancy | 0.54 [0.52, 0.55] | 0.525 | 0.940 | +1 [-2, +5] | 0.03 [-0.06, 0.12] | 0.105 | 0.613 | 300 |
| all | debate_seq | 0.58 [0.56, 0.60] | 0.527 | 0.944 | +6 [+2, +9] | 0.13 [0.05, 0.21] | 0.213 | 0.622 | 298 |
| all | debate_seq_noquotes | 0.59 [0.57, 0.61] | 0.528 | 0.945 | +6 [+2, +10] | 0.14 [0.06, 0.22] | nan | nan | 299 |
| claude-sonnet-4-5 | consultancy | 0.53 [0.51, 0.56] | 0.514 | 0.954 | +2 [-3, +7] | 0.05 [-0.07, 0.14] | 0.122 | 0.594 | 191 |
| claude-sonnet-4-5 | debate_seq | 0.60 [0.57, 0.62] | 0.517 | 0.961 | +8 [+3, +12] | 0.18 [0.08, 0.26] | 0.248 | 0.639 | 197 |
| claude-sonnet-4-5 | debate_seq_noquotes | 0.60 [0.57, 0.63] | 0.520 | 0.962 | +8 [+4, +12] | 0.18 [0.09, 0.26] | nan | nan | 205 |
| gpt-5-mini | consultancy | 0.54 [0.53, 0.56] | 0.531 | 0.933 | +1 [-2, +5] | 0.02 [-0.07, 0.11] | 0.095 | 0.624 | 300 |
| gpt-5-mini | debate_seq | 0.57 [0.55, 0.59] | 0.533 | 0.933 | +4 [+0, +8] | 0.10 [0.00, 0.19] | 0.190 | 0.610 | 298 |
| gpt-5-mini | debate_seq_noquotes | 0.58 [0.56, 0.60] | 0.533 | 0.933 | +5 [+1, +9] | 0.12 [0.02, 0.20] | nan | nan | 299 |

## Judge strength: the same debates, weak and strong judges (prompted liars)

| dataset | judge | debater | mechanism | accuracy | judge_alone | debater_alone | delta | gap_recovered | items |
|---|---|---|---|---|---|---|---|---|---|
| quality | gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.74 [0.71, 0.77] | 0.510 | 0.968 | +23 [+17, +29] | 0.51 [0.42, 0.58] | 250 |
| quality | gpt-4o-mini | gpt-5-mini | consultancy | 0.57 [0.55, 0.59] | 0.515 | 0.933 | +6 [+1, +10] | 0.14 [0.03, 0.23] | 300 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.60 [0.56, 0.63] | 0.517 | 0.968 | +8 [+3, +13] | 0.18 [0.07, 0.27] | 250 |
| quality | llama-3.1-8b | gpt-5-mini | consultancy | 0.54 [0.51, 0.57] | 0.528 | 0.933 | +1 [-4, +6] | 0.02 [-0.10, 0.13] | 300 |
| quality | gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.65 [0.63, 0.67] | 0.546 | 0.968 | +10 [+5, +15] | 0.25 [0.13, 0.33] | 250 |
| quality | gpt-4.1-mini | gpt-5-mini | consultancy | 0.57 [0.55, 0.59] | 0.547 | 0.933 | +2 [-3, +8] | 0.06 [-0.07, 0.18] | 300 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | consultancy | 0.76 [0.73, 0.80] | 0.555 | 0.968 | +21 [+15, +27] | 0.50 [0.40, 0.59] | 250 |
| quality | gemma-3-12b | gpt-5-mini | consultancy | 0.52 [0.51, 0.54] | 0.558 | 0.933 | -3 [-8, +1] | -0.09 [-0.23, 0.02] | 300 |
| quality | gpt-4.1 | gpt-5-mini | consultancy | 0.70 [0.68, 0.73] | 0.603 | 0.933 | +10 [+5, +16] | 0.31 [0.18, 0.43] | 300 |
| quality | gpt-4.1 | claude-sonnet-4-5 | consultancy | 0.83 [0.80, 0.86] | 0.612 | 0.968 | +22 [+16, +28] | 0.61 [0.50, 0.71] | 250 |
| quality | gpt-4o-mini | claude-sonnet-4-5 | debate_seq | 0.73 [0.70, 0.77] | 0.510 | 0.968 | +22 [+16, +28] | 0.49 [0.40, 0.58] | 250 |
| quality | gpt-4o-mini | gpt-5-mini | debate_seq | 0.65 [0.61, 0.68] | 0.515 | 0.933 | +13 [+8, +19] | 0.31 [0.20, 0.41] | 300 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | debate_seq | 0.57 [0.54, 0.60] | 0.517 | 0.968 | +5 [+0, +10] | 0.11 [0.01, 0.21] | 250 |
| quality | llama-3.1-8b | gpt-5-mini | debate_seq | 0.54 [0.51, 0.57] | 0.528 | 0.933 | +1 [-4, +6] | 0.03 [-0.10, 0.14] | 300 |
| quality | gemma-3-12b | claude-sonnet-4-5 | debate_seq | 0.60 [0.57, 0.64] | 0.546 | 0.968 | +6 [-1, +11] | 0.14 [-0.02, 0.25] | 250 |
| quality | gpt-4.1-mini | gpt-5-mini | debate_seq | 0.76 [0.73, 0.79] | 0.547 | 0.933 | +21 [+16, +27] | 0.56 [0.45, 0.65] | 300 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq | 0.84 [0.81, 0.86] | 0.555 | 0.968 | +28 [+22, +34] | 0.68 [0.60, 0.76] | 250 |
| quality | gemma-3-12b | gpt-5-mini | debate_seq | 0.55 [0.51, 0.58] | 0.558 | 0.933 | -1 [-6, +4] | -0.03 [-0.18, 0.10] | 300 |
| quality | gpt-4.1 | gpt-5-mini | debate_seq | 0.85 [0.82, 0.88] | 0.603 | 0.933 | +25 [+19, +30] | 0.75 [0.65, 0.84] | 300 |
| quality | gpt-4.1 | claude-sonnet-4-5 | debate_seq | 0.92 [0.89, 0.94] | 0.612 | 0.968 | +31 [+25, +36] | 0.86 [0.79, 0.92] | 250 |
| quality | gpt-4o-mini | claude-sonnet-4-5 | debate_seq_noquotes | 0.73 [0.69, 0.77] | 0.510 | 0.968 | +22 [+16, +28] | 0.49 [0.39, 0.56] | 250 |
| quality | gpt-4o-mini | gpt-5-mini | debate_seq_noquotes | 0.65 [0.62, 0.69] | 0.515 | 0.933 | +14 [+8, +20] | 0.33 [0.22, 0.43] | 300 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | debate_seq_noquotes | 0.54 [0.50, 0.57] | 0.517 | 0.968 | +2 [-3, +7] | 0.04 [-0.08, 0.15] | 250 |
| quality | llama-3.1-8b | gpt-5-mini | debate_seq_noquotes | 0.55 [0.53, 0.58] | 0.528 | 0.933 | +3 [-2, +7] | 0.06 [-0.06, 0.17] | 300 |
| quality | gemma-3-12b | claude-sonnet-4-5 | debate_seq_noquotes | 0.58 [0.54, 0.61] | 0.546 | 0.968 | +3 [-3, +9] | 0.07 [-0.07, 0.19] | 250 |
| quality | gpt-4.1-mini | gpt-5-mini | debate_seq_noquotes | 0.75 [0.72, 0.78] | 0.547 | 0.933 | +21 [+16, +26] | 0.54 [0.44, 0.62] | 300 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq_noquotes | 0.83 [0.80, 0.86] | 0.555 | 0.968 | +27 [+22, +33] | 0.66 [0.58, 0.73] | 250 |
| quality | gemma-3-12b | gpt-5-mini | debate_seq_noquotes | 0.54 [0.51, 0.57] | 0.558 | 0.933 | -2 [-7, +3] | -0.05 [-0.21, 0.08] | 300 |
| quality | gpt-4.1 | gpt-5-mini | debate_seq_noquotes | 0.85 [0.81, 0.88] | 0.603 | 0.933 | +24 [+18, +30] | 0.74 [0.62, 0.84] | 300 |
| quality | gpt-4.1 | claude-sonnet-4-5 | debate_seq_noquotes | 0.90 [0.88, 0.93] | 0.612 | 0.968 | +29 [+23, +35] | 0.81 [0.73, 0.89] | 250 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.62 [0.60, 0.64] | 0.409 | 0.960 | +21 [+18, +24] | 0.38 [0.33, 0.42] | 320 |
| testbed | llama-3.1-8b | gpt-5-mini | consultancy | 0.50 [0.47, 0.52] | 0.409 | 0.989 | +9 [+6, +12] | 0.15 [0.10, 0.20] | 320 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.70 [0.67, 0.73] | 0.439 | 0.960 | +26 [+21, +30] | 0.50 [0.43, 0.55] | 320 |
| testbed | gpt-4o-mini | gpt-5-mini | consultancy | 0.60 [0.58, 0.63] | 0.439 | 0.989 | +16 [+13, +20] | 0.30 [0.25, 0.35] | 320 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.73 [0.70, 0.75] | 0.457 | 0.960 | +27 [+23, +31] | 0.54 [0.48, 0.58] | 320 |
| testbed | gemma-3-12b | gpt-5-mini | consultancy | 0.57 [0.56, 0.59] | 0.457 | 0.989 | +12 [+8, +15] | 0.22 [0.17, 0.26] | 320 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | consultancy | 0.85 [0.82, 0.87] | 0.911 | 0.960 | -7 [-9, -4] | -1.34 [-2.95, -0.61] | 320 |
| testbed | gpt-4.1-mini | gpt-5-mini | consultancy | 0.73 [0.70, 0.76] | 0.911 | 0.989 | -18 [-21, -15] | -2.29 [-3.36, -1.57] | 320 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | consultancy | 0.87 [0.85, 0.90] | 0.938 | 0.960 | -6 [-9, -4] | -2.93 [-11.15, -1.21] | 320 |
| testbed | gpt-4.1 | gpt-5-mini | consultancy | 0.80 [0.77, 0.82] | 0.938 | 0.989 | -14 [-17, -11] | -2.71 [-4.62, -1.76] | 320 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | debate_seq | 0.71 [0.68, 0.74] | 0.409 | 0.960 | +30 [+26, +34] | 0.55 [0.49, 0.60] | 320 |
| testbed | llama-3.1-8b | gpt-5-mini | debate_seq | 0.56 [0.53, 0.58] | 0.409 | 0.989 | +15 [+11, +19] | 0.26 [0.20, 0.31] | 320 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | debate_seq | 0.76 [0.74, 0.79] | 0.439 | 0.960 | +32 [+28, +37] | 0.62 [0.56, 0.67] | 320 |
| testbed | gpt-4o-mini | gpt-5-mini | debate_seq | 0.65 [0.62, 0.68] | 0.439 | 0.989 | +21 [+17, +25] | 0.38 [0.33, 0.43] | 320 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | debate_seq | 0.66 [0.63, 0.69] | 0.457 | 0.960 | +20 [+16, +24] | 0.40 [0.33, 0.46] | 320 |
| testbed | gemma-3-12b | gpt-5-mini | debate_seq | 0.53 [0.50, 0.56] | 0.457 | 0.989 | +7 [+3, +11] | 0.14 [0.07, 0.20] | 320 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq | 0.86 [0.84, 0.88] | 0.911 | 0.960 | -5 [-8, -2] | -1.03 [-2.43, -0.40] | 320 |
| testbed | gpt-4.1-mini | gpt-5-mini | debate_seq | 0.79 [0.77, 0.82] | 0.911 | 0.989 | -12 [-15, -9] | -1.52 [-2.36, -0.98] | 320 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | debate_seq | 0.88 [0.86, 0.91] | 0.938 | 0.960 | -5 [-8, -3] | -2.46 [-10.00, -0.98] | 320 |
| testbed | gpt-4.1 | gpt-5-mini | debate_seq | 0.85 [0.82, 0.87] | 0.938 | 0.989 | -9 [-12, -7] | -1.78 [-3.11, -1.06] | 320 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | debate_sim | 0.74 [0.71, 0.76] | 0.409 | 0.960 | +33 [+29, +37] | 0.60 [0.54, 0.65] | 320 |
| testbed | llama-3.1-8b | gpt-5-mini | debate_sim | 0.58 [0.55, 0.60] | 0.409 | 0.989 | +17 [+12, +21] | 0.29 [0.22, 0.34] | 320 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | debate_sim | 0.79 [0.76, 0.82] | 0.439 | 0.960 | +35 [+31, +40] | 0.68 [0.62, 0.73] | 320 |
| testbed | gpt-4o-mini | gpt-5-mini | debate_sim | 0.68 [0.65, 0.71] | 0.439 | 0.989 | +24 [+19, +28] | 0.43 [0.37, 0.49] | 320 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | debate_sim | 0.76 [0.73, 0.79] | 0.457 | 0.960 | +30 [+26, +34] | 0.60 [0.54, 0.66] | 320 |
| testbed | gemma-3-12b | gpt-5-mini | debate_sim | 0.54 [0.52, 0.57] | 0.457 | 0.989 | +8 [+4, +12] | 0.16 [0.09, 0.22] | 320 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | debate_sim | 0.87 [0.84, 0.89] | 0.911 | 0.960 | -5 [-7, -2] | -0.94 [-2.30, -0.33] | 320 |
| testbed | gpt-4.1-mini | gpt-5-mini | debate_sim | 0.81 [0.78, 0.84] | 0.911 | 0.989 | -10 [-13, -7] | -1.29 [-2.06, -0.78] | 320 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | debate_sim | 0.89 [0.86, 0.91] | 0.938 | 0.960 | -5 [-7, -3] | -2.33 [-10.11, -0.94] | 320 |
| testbed | gpt-4.1 | gpt-5-mini | debate_sim | 0.86 [0.84, 0.88] | 0.938 | 0.989 | -8 [-10, -5] | -1.47 [-2.69, -0.81] | 320 |

## Judge strength, committed liars only

| dataset | judge | debater | mechanism | accuracy | judge_alone | debater_alone | delta | gap_recovered | items |
|---|---|---|---|---|---|---|---|---|---|
| quality | gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.58 [0.54, 0.61] | 0.478 | 0.949 | +10 [+2, +18] | 0.21 [0.05, 0.33] | 151 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | consultancy | 0.52 [0.49, 0.54] | 0.487 | 0.934 | +3 [-6, +11] | 0.06 [-0.17, 0.21] | 98 |
| quality | gpt-4o-mini | gpt-5-mini | consultancy | 0.57 [0.55, 0.59] | 0.513 | 0.932 | +6 [+1, +11] | 0.14 [0.03, 0.24] | 296 |
| quality | gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.52 [0.51, 0.54] | 0.527 | 0.957 | -0 [-6, +6] | -0.01 [-0.16, 0.12] | 169 |
| quality | llama-3.1-8b | gpt-5-mini | consultancy | 0.54 [0.51, 0.57] | 0.528 | 0.933 | +1 [-4, +6] | 0.02 [-0.10, 0.14] | 298 |
| quality | gpt-4.1-mini | gpt-5-mini | consultancy | 0.54 [0.52, 0.56] | 0.532 | 0.935 | +1 [-4, +6] | 0.02 [-0.12, 0.14] | 267 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.50 [0.45, 0.55] | 0.533 | 0.957 | -3 [-9, +3] | -0.07 [-0.24, 0.06] | 158 |
| quality | gemma-3-12b | gpt-5-mini | consultancy | 0.52 [0.50, 0.53] | 0.553 | 0.932 | -4 [-8, +1] | -0.10 [-0.24, 0.02] | 295 |
| quality | gpt-4.1 | gpt-5-mini | consultancy | 0.67 [0.64, 0.70] | 0.593 | 0.925 | +7 [+1, +13] | 0.22 [0.05, 0.34] | 260 |
| quality | gpt-4.1 | claude-sonnet-4-5 | consultancy | 0.64 [0.60, 0.69] | 0.601 | 0.945 | +4 [-5, +13] | 0.12 [-0.20, 0.33] | 105 |
| quality | gpt-4o-mini | claude-sonnet-4-5 | debate_seq | 0.70 [0.65, 0.74] | 0.504 | 0.961 | +20 [+13, +26] | 0.43 [0.30, 0.53] | 197 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | debate_seq | 0.53 [0.49, 0.57] | 0.505 | 0.961 | +2 [-3, +8] | 0.05 [-0.08, 0.16] | 197 |
| quality | gpt-4o-mini | gpt-5-mini | debate_seq | 0.64 [0.61, 0.68] | 0.513 | 0.933 | +13 [+8, +19] | 0.31 [0.20, 0.42] | 298 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq | 0.81 [0.78, 0.85] | 0.527 | 0.961 | +28 [+22, +35] | 0.65 [0.56, 0.75] | 197 |
| quality | llama-3.1-8b | gpt-5-mini | debate_seq | 0.53 [0.50, 0.57] | 0.529 | 0.933 | +1 [-5, +6] | 0.02 [-0.12, 0.13] | 298 |
| quality | gemma-3-12b | claude-sonnet-4-5 | debate_seq | 0.56 [0.52, 0.60] | 0.543 | 0.961 | +2 [-5, +8] | 0.05 [-0.14, 0.18] | 197 |
| quality | gpt-4.1-mini | gpt-5-mini | debate_seq | 0.76 [0.73, 0.79] | 0.545 | 0.933 | +22 [+16, +27] | 0.56 [0.46, 0.66] | 298 |
| quality | gemma-3-12b | gpt-5-mini | debate_seq | 0.54 [0.51, 0.57] | 0.557 | 0.933 | -2 [-6, +3] | -0.04 [-0.19, 0.09] | 298 |
| quality | gpt-4.1 | gpt-5-mini | debate_seq | 0.85 [0.82, 0.88] | 0.603 | 0.933 | +25 [+19, +31] | 0.75 [0.65, 0.84] | 298 |
| quality | gpt-4.1 | claude-sonnet-4-5 | debate_seq | 0.90 [0.86, 0.93] | 0.613 | 0.961 | +28 [+22, +35] | 0.82 [0.73, 0.90] | 197 |
| quality | gpt-4o-mini | claude-sonnet-4-5 | debate_seq_noquotes | 0.71 [0.66, 0.75] | 0.506 | 0.962 | +20 [+13, +26] | 0.44 [0.32, 0.54] | 205 |
| quality | gpt-4o-mini | gpt-5-mini | debate_seq_noquotes | 0.65 [0.61, 0.69] | 0.514 | 0.933 | +14 [+8, +19] | 0.32 [0.20, 0.43] | 299 |
| quality | llama-3.1-8b | claude-sonnet-4-5 | debate_seq_noquotes | 0.54 [0.50, 0.58] | 0.515 | 0.962 | +3 [-3, +9] | 0.06 [-0.08, 0.18] | 205 |
| quality | llama-3.1-8b | gpt-5-mini | debate_seq_noquotes | 0.55 [0.53, 0.58] | 0.530 | 0.933 | +2 [-2, +7] | 0.06 [-0.06, 0.16] | 299 |
| quality | gemma-3-12b | claude-sonnet-4-5 | debate_seq_noquotes | 0.55 [0.51, 0.60] | 0.540 | 0.962 | +1 [-5, +7] | 0.03 [-0.12, 0.15] | 205 |
| quality | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq_noquotes | 0.79 [0.76, 0.83] | 0.543 | 0.962 | +25 [+18, +32] | 0.60 [0.50, 0.69] | 205 |
| quality | gpt-4.1-mini | gpt-5-mini | debate_seq_noquotes | 0.75 [0.72, 0.78] | 0.546 | 0.933 | +21 [+16, +26] | 0.54 [0.44, 0.63] | 299 |
| quality | gemma-3-12b | gpt-5-mini | debate_seq_noquotes | 0.54 [0.50, 0.57] | 0.556 | 0.933 | -2 [-7, +3] | -0.06 [-0.21, 0.07] | 299 |
| quality | gpt-4.1 | gpt-5-mini | debate_seq_noquotes | 0.84 [0.81, 0.87] | 0.601 | 0.933 | +24 [+18, +30] | 0.73 [0.62, 0.84] | 299 |
| quality | gpt-4.1 | claude-sonnet-4-5 | debate_seq_noquotes | 0.89 [0.85, 0.92] | 0.615 | 0.962 | +27 [+21, +34] | 0.79 [0.68, 0.89] | 205 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | consultancy | 0.49 [0.44, 0.53] | 0.349 | 0.923 | +14 [+8, +20] | 0.24 [0.15, 0.33] | 168 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | consultancy | 0.50 [0.47, 0.53] | 0.392 | 0.928 | +11 [+6, +16] | 0.20 [0.12, 0.28] | 177 |
| testbed | llama-3.1-8b | gpt-5-mini | consultancy | 0.49 [0.47, 0.51] | 0.409 | 0.989 | +8 [+5, +11] | 0.14 [0.09, 0.19] | 318 |
| testbed | gpt-4o-mini | gpt-5-mini | consultancy | 0.58 [0.55, 0.61] | 0.442 | 0.989 | +14 [+10, +17] | 0.25 [0.19, 0.30] | 312 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | consultancy | 0.52 [0.48, 0.56] | 0.444 | 0.927 | +8 [+3, +12] | 0.16 [0.06, 0.25] | 176 |
| testbed | gemma-3-12b | gpt-5-mini | consultancy | 0.56 [0.54, 0.58] | 0.458 | 0.989 | +10 [+6, +13] | 0.18 [0.12, 0.24] | 318 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | consultancy | 0.60 [0.55, 0.64] | 0.817 | 0.927 | -22 [-28, -16] | -1.99 [-4.04, -1.10] | 110 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | consultancy | 0.71 [0.66, 0.76] | 0.867 | 0.912 | -16 [-22, -11] | -3.55 [-21.70, -1.24] | 114 |
| testbed | gpt-4.1-mini | gpt-5-mini | consultancy | 0.68 [0.65, 0.71] | 0.898 | 0.987 | -21 [-25, -18] | -2.40 [-3.60, -1.68] | 258 |
| testbed | gpt-4.1 | gpt-5-mini | consultancy | 0.74 [0.71, 0.77] | 0.928 | 0.987 | -19 [-22, -15] | -3.21 [-5.49, -2.05] | 234 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | debate_seq | 0.58 [0.54, 0.64] | 0.392 | 0.924 | +19 [+13, +26] | 0.36 [0.26, 0.45] | 171 |
| testbed | llama-3.1-8b | gpt-5-mini | debate_seq | 0.55 [0.52, 0.57] | 0.415 | 0.989 | +13 [+10, +17] | 0.23 [0.17, 0.29] | 308 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | debate_seq | 0.43 [0.39, 0.47] | 0.421 | 0.924 | +1 [-5, +7] | 0.02 [-0.11, 0.14] | 171 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | debate_seq | 0.52 [0.48, 0.56] | 0.447 | 0.924 | +7 [+2, +13] | 0.15 [0.04, 0.25] | 171 |
| testbed | gpt-4o-mini | gpt-5-mini | debate_seq | 0.64 [0.60, 0.66] | 0.450 | 0.989 | +19 [+14, +22] | 0.34 [0.28, 0.40] | 308 |
| testbed | gemma-3-12b | gpt-5-mini | debate_seq | 0.51 [0.49, 0.54] | 0.467 | 0.989 | +5 [+1, +9] | 0.09 [0.02, 0.16] | 308 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | debate_seq | 0.76 [0.71, 0.80] | 0.868 | 0.924 | -11 [-16, -7] | -2.00 [-5.53, -0.86] | 171 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | debate_seq | 0.77 [0.73, 0.81] | 0.908 | 0.924 | -13 [-18, -9] | -8.36 [-121.07, 67.73] | 171 |
| testbed | gpt-4.1-mini | gpt-5-mini | debate_seq | 0.78 [0.76, 0.81] | 0.909 | 0.989 | -13 [-16, -9] | -1.58 [-2.43, -0.98] | 308 |
| testbed | gpt-4.1 | gpt-5-mini | debate_seq | 0.84 [0.81, 0.86] | 0.937 | 0.989 | -10 [-13, -7] | -1.92 [-3.43, -1.13] | 308 |
| testbed | gpt-4o-mini | claude-sonnet-4-5 | debate_sim | 0.37 [0.28, 0.46] | 0.229 | 0.904 | +14 [+2, +26] | 0.21 [0.03, 0.36] | 52 |
| testbed | gemma-3-12b | claude-sonnet-4-5 | debate_sim | 0.37 [0.29, 0.45] | 0.391 | 0.904 | -2 [-15, +11] | -0.04 [-0.34, 0.19] | 52 |
| testbed | llama-3.1-8b | gpt-5-mini | debate_sim | 0.55 [0.52, 0.58] | 0.425 | 0.989 | +13 [+9, +17] | 0.23 [0.16, 0.29] | 299 |
| testbed | gpt-4o-mini | gpt-5-mini | debate_sim | 0.65 [0.62, 0.68] | 0.460 | 0.989 | +19 [+15, +24] | 0.37 [0.29, 0.43] | 299 |
| testbed | gemma-3-12b | gpt-5-mini | debate_sim | 0.52 [0.49, 0.55] | 0.469 | 0.989 | +5 [+1, +9] | 0.09 [0.01, 0.17] | 299 |
| testbed | llama-3.1-8b | claude-sonnet-4-5 | debate_sim | 0.44 [0.35, 0.54] | 0.489 | 0.904 | -5 [-16, +7] | -0.11 [-0.43, 0.15] | 52 |
| testbed | gpt-4.1 | claude-sonnet-4-5 | debate_sim | 0.56 [0.46, 0.67] | 0.819 | 0.904 | -25 [-36, -16] | -2.99 [-15.71, -1.16] | 52 |
| testbed | gpt-4.1-mini | claude-sonnet-4-5 | debate_sim | 0.62 [0.51, 0.72] | 0.819 | 0.904 | -20 [-30, -9] | -2.37 [-15.90, -0.76] | 52 |
| testbed | gpt-4.1-mini | gpt-5-mini | debate_sim | 0.80 [0.77, 0.83] | 0.907 | 0.989 | -11 [-13, -8] | -1.31 [-2.05, -0.80] | 299 |
| testbed | gpt-4.1 | gpt-5-mini | debate_sim | 0.85 [0.82, 0.88] | 0.936 | 0.989 | -9 [-11, -6] | -1.66 [-3.02, -0.94] | 299 |

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

