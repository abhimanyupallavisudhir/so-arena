# ASD after a calibration map

Each judge's stated probability goes through a Platt map (logit p' = a + b logit p, b >= 0), fitted per judge and protocol and cross-fitted over items. `b` above 1 means the judge was underconfident, below 1 overconfident. All prompted liars; Brier ASD unless stated.

## Model panel (300 unselected items): is the judge less confident with an argument?

| judge | condition | accuracy | mean abs(p - 0.5) | a | b |
|---|---|---|---|---|---|
| deepseek-v3.2 | consultancy (one argument) | 0.875 | 0.391 | -0.25 | 0.77 |
| deepseek-v3.2 | debate, simultaneous | 0.874 | 0.388 | +0.00 | 0.98 |
| deepseek-v3.2 | judge, no argument | 0.882 | 0.419 | +0.00 | 0.81 |
| gemma-3-12b | consultancy (one argument) | 0.784 | 0.371 | -0.24 | 0.28 |
| gemma-3-12b | debate, simultaneous | 0.625 | 0.348 | +0.00 | 0.29 |
| gemma-3-12b | judge, no argument | 0.633 | 0.376 | +0.00 | 0.17 |
| gpt-4.1 | consultancy (one argument) | 0.922 | 0.466 | -0.59 | 0.47 |
| gpt-4.1 | debate, simultaneous | 0.844 | 0.464 | +0.00 | 0.49 |
| gpt-4.1 | judge, no argument | 0.843 | 0.465 | +0.00 | 0.51 |
| gpt-4.1-mini | consultancy (one argument) | 0.927 | 0.432 | -0.68 | 0.53 |
| gpt-4.1-mini | debate, simultaneous | 0.821 | 0.394 | +0.00 | 0.69 |
| gpt-4.1-mini | judge, no argument | 0.858 | 0.411 | +0.00 | 0.68 |
| gpt-4o-mini | consultancy (one argument) | 0.601 | 0.305 | -0.08 | 0.32 |
| gpt-4o-mini | debate, simultaneous | 0.677 | 0.319 | +0.00 | 0.55 |
| gpt-4o-mini | judge, no argument | 0.553 | 0.372 | +0.00 | 0.11 |
| llama-3.1-8b | consultancy (one argument) | 0.426 | 0.219 | +0.01 | 0.06 |
| llama-3.1-8b | debate, simultaneous | 0.607 | 0.268 | +0.00 | 0.32 |
| llama-3.1-8b | judge, no argument | 0.530 | 0.306 | +0.00 | 0.04 |
| llama-3.2-3b | consultancy (one argument) | 0.487 | 0.209 | -0.00 | 0.02 |
| llama-3.2-3b | debate, simultaneous | 0.565 | 0.315 | +0.00 | 0.12 |
| llama-3.2-3b | judge, no argument | 0.497 | 0.288 | +0.00 | 0.00 |
| mistral-small-4 | consultancy (one argument) | 0.862 | 0.430 | -0.43 | 0.38 |
| mistral-small-4 | debate, simultaneous | 0.761 | 0.421 | +0.00 | 0.44 |
| mistral-small-4 | judge, no argument | 0.720 | 0.441 | +0.00 | 0.30 |
| qwen3-14b | consultancy (one argument) | 0.856 | 0.349 | -0.40 | 0.33 |
| qwen3-14b | debate, simultaneous | 0.701 | 0.329 | +0.00 | 0.44 |
| qwen3-14b | judge, no argument | 0.630 | 0.330 | +0.00 | 0.24 |

## Model panel: gain over the same judge with no argument

| judge | protocol | raw ASD | raw gain | calibrated ASD | calibrated gain |
|---|---|---|---|---|---|
| all | consultancy (one argument) | +0.324 | -0.015 [-0.040, +0.011] | +0.255 | +0.016 [+0.005, +0.027] |
| all | debate, simultaneous | +0.399 | +0.060 [+0.031, +0.090] | +0.307 | +0.067 [+0.053, +0.081] |
| deepseek-v3.2 | consultancy (one argument) | +0.608 | -0.090 [-0.123, -0.058] | +0.578 | -0.077 [-0.107, -0.048] |
| deepseek-v3.2 | debate, simultaneous | +0.655 | -0.044 [-0.075, -0.012] | +0.647 | -0.008 [-0.037, +0.020] |
| gemma-3-12b | consultancy (one argument) | +0.265 | +0.055 [-0.014, +0.123] | +0.150 | +0.079 [+0.055, +0.102] |
| gemma-3-12b | debate, simultaneous | +0.246 | +0.036 [-0.037, +0.106] | +0.132 | +0.061 [+0.035, +0.085] |
| gpt-4.1 | consultancy (one argument) | +0.599 | -0.074 [-0.117, -0.031] | +0.510 | -0.028 [-0.059, +0.003] |
| gpt-4.1 | debate, simultaneous | +0.666 | -0.007 [-0.050, +0.036] | +0.525 | -0.013 [-0.045, +0.019] |
| gpt-4.1-mini | consultancy (one argument) | +0.503 | -0.092 [-0.132, -0.047] | +0.441 | -0.064 [-0.096, -0.027] |
| gpt-4.1-mini | debate, simultaneous | +0.556 | -0.040 [-0.083, +0.009] | +0.477 | -0.028 [-0.063, +0.013] |
| gpt-4o-mini | consultancy (one argument) | +0.208 | +0.073 [+0.013, +0.133] | +0.113 | +0.087 [+0.069, +0.106] |
| gpt-4o-mini | debate, simultaneous | +0.305 | +0.171 [+0.099, +0.243] | +0.224 | +0.198 [+0.170, +0.225] |
| llama-3.1-8b | consultancy (one argument) | +0.029 | -0.015 [-0.075, +0.045] | +0.003 | -0.000 [-0.005, +0.005] |
| llama-3.1-8b | debate, simultaneous | +0.172 | +0.128 [+0.062, +0.192] | +0.091 | +0.088 [+0.075, +0.101] |
| llama-3.2-3b | consultancy (one argument) | +0.013 | +0.014 [-0.049, +0.077] | +0.000 | +0.002 [+0.000, +0.004] |
| llama-3.2-3b | debate, simultaneous | +0.121 | +0.122 [+0.056, +0.186] | +0.031 | +0.033 [+0.027, +0.039] |
| mistral-small-4 | consultancy (one argument) | +0.432 | -0.011 [-0.072, +0.053] | +0.321 | +0.073 [+0.041, +0.108] |
| mistral-small-4 | debate, simultaneous | +0.501 | +0.058 [-0.003, +0.124] | +0.366 | +0.118 [+0.083, +0.156] |
| qwen3-14b | consultancy (one argument) | +0.256 | +0.004 [-0.052, +0.063] | +0.180 | +0.068 [+0.042, +0.095] |
| qwen3-14b | debate, simultaneous | +0.369 | +0.116 [+0.052, +0.183] | +0.268 | +0.156 [+0.122, +0.190] |

## Appendix: the panel with committed liars only (gpt-5 audit)

| judge | protocol | raw gain | calibrated gain |
|---|---|---|---|
| all | consultancy (one argument) | -0.066 [-0.090, -0.041] | -0.046 [-0.055, -0.036] |
| all | debate, simultaneous | -0.009 [-0.036, +0.019] | -0.015 [-0.025, -0.003] |
| deepseek-v3.2 | consultancy (one argument) | -0.102 [-0.135, -0.070] | -0.108 [-0.138, -0.078] |
| deepseek-v3.2 | debate, simultaneous | -0.056 [-0.088, -0.023] | -0.027 [-0.058, +0.002] |
| gemma-3-12b | consultancy (one argument) | -0.024 [-0.088, +0.042] | -0.008 [-0.029, +0.014] |
| gemma-3-12b | debate, simultaneous | -0.092 [-0.166, -0.024] | -0.048 [-0.072, -0.025] |
| gpt-4.1 | consultancy (one argument) | -0.098 [-0.142, -0.055] | -0.093 [-0.124, -0.061] |
| gpt-4.1 | debate, simultaneous | -0.023 [-0.067, +0.021] | -0.079 [-0.111, -0.046] |
| gpt-4.1-mini | consultancy (one argument) | -0.135 [-0.175, -0.091] | -0.137 [-0.169, -0.101] |
| gpt-4.1-mini | debate, simultaneous | -0.084 [-0.128, -0.034] | -0.116 [-0.153, -0.074] |
| gpt-4o-mini | consultancy (one argument) | -0.003 [-0.065, +0.058] | +0.011 [-0.002, +0.025] |
| gpt-4o-mini | debate, simultaneous | +0.076 [+0.010, +0.145] | +0.065 [+0.049, +0.082] |
| llama-3.1-8b | consultancy (one argument) | -0.023 [-0.085, +0.038] | -0.002 [-0.007, +0.003] |
| llama-3.1-8b | debate, simultaneous | +0.042 [-0.022, +0.108] | +0.016 [+0.009, +0.023] |
| llama-3.2-3b | consultancy (one argument) | -0.007 [-0.067, +0.055] | +0.001 [-0.000, +0.003] |
| llama-3.2-3b | debate, simultaneous | +0.033 [-0.032, +0.097] | +0.002 [+0.000, +0.005] |
| mistral-small-4 | consultancy (one argument) | -0.067 [-0.127, -0.003] | -0.021 [-0.053, +0.013] |
| mistral-small-4 | debate, simultaneous | +0.012 [-0.050, +0.079] | +0.025 [-0.008, +0.062] |
| qwen3-14b | consultancy (one argument) | -0.137 [-0.196, -0.074] | -0.057 [-0.084, -0.029] |
| qwen3-14b | debate, simultaneous | +0.013 [-0.053, +0.081] | +0.030 [-0.001, +0.062] |

## Protocol grid (927 gap-selected items, gpt-5-mini)

`own map`: fitted on these items, so it learns the item selection. `panel map`: fitted on the panel's unselected items and applied here.

| protocol | judge | treatment | ASD, Brier | ASD, log | gain over no argument, Brier |
|---|---|---|---|---|---|
| consultancy | all | own map | +0.039 [+0.034, +0.044] | +0.084 | +0.039 [+0.034, +0.044] |
| consultancy | all | panel map | +0.049 [+0.042, +0.055] | +0.108 | +0.064 [+0.058, +0.070] |
| consultancy | all | raw | +0.080 [+0.065, +0.094] | +0.378 | +0.188 [+0.166, +0.209] |
| consultancy | gemma-3-12b | own map | +0.071 [+0.061, +0.080] | +0.156 | +0.071 [+0.061, +0.080] |
| consultancy | gemma-3-12b | panel map | +0.086 [+0.075, +0.098] | +0.195 | +0.100 [+0.085, +0.114] |
| consultancy | gemma-3-12b | raw | +0.140 [+0.119, +0.161] | +0.912 | +0.199 [+0.163, +0.235] |
| consultancy | gpt-4o-mini | own map | +0.046 [+0.038, +0.054] | +0.097 | +0.046 [+0.038, +0.054] |
| consultancy | gpt-4o-mini | panel map | +0.061 [+0.050, +0.071] | +0.132 | +0.080 [+0.071, +0.090] |
| consultancy | gpt-4o-mini | raw | +0.119 [+0.099, +0.139] | +0.482 | +0.228 [+0.194, +0.261] |
| consultancy | llama-3.1-8b | own map | +0.000 [+0.000, +0.000] | +0.000 | +0.000 [+0.000, +0.000] |
| consultancy | llama-3.1-8b | panel map | -0.002 [-0.004, +0.001] | -0.003 | +0.012 [+0.009, +0.015] |
| consultancy | llama-3.1-8b | raw | -0.020 [-0.038, -0.003] | -0.260 | +0.136 [+0.106, +0.165] |
| debate, sequential | all | own map | +0.068 [+0.061, +0.074] | +0.147 | +0.068 [+0.061, +0.074] |
| debate, sequential | all | panel map | +0.095 [+0.086, +0.104] | +0.219 | +0.111 [+0.101, +0.121] |
| debate, sequential | all | raw | +0.153 [+0.137, +0.169] | +0.742 | +0.260 [+0.232, +0.287] |
| debate, sequential | gemma-3-12b | own map | +0.023 [+0.019, +0.028] | +0.047 | +0.023 [+0.019, +0.028] |
| debate, sequential | gemma-3-12b | panel map | +0.054 [+0.043, +0.064] | +0.120 | +0.067 [+0.050, +0.083] |
| debate, sequential | gemma-3-12b | raw | +0.096 [+0.072, +0.120] | +0.638 | +0.154 [+0.111, +0.196] |
| debate, sequential | gpt-4o-mini | own map | +0.138 [+0.124, +0.151] | +0.305 | +0.138 [+0.124, +0.151] |
| debate, sequential | gpt-4o-mini | panel map | +0.164 [+0.148, +0.180] | +0.382 | +0.183 [+0.168, +0.198] |
| debate, sequential | gpt-4o-mini | raw | +0.236 [+0.212, +0.259] | +0.792 | +0.345 [+0.307, +0.382] |
| debate, sequential | llama-3.1-8b | own map | +0.042 [+0.035, +0.049] | +0.088 | +0.042 [+0.035, +0.049] |
| debate, sequential | llama-3.1-8b | panel map | +0.069 [+0.058, +0.080] | +0.156 | +0.083 [+0.071, +0.094] |
| debate, sequential | llama-3.1-8b | raw | +0.126 [+0.106, +0.147] | +0.798 | +0.282 [+0.247, +0.317] |
| debate, simultaneous | all | own map | +0.080 [+0.073, +0.088] | +0.176 | +0.080 [+0.073, +0.088] |
| debate, simultaneous | all | panel map | +0.105 [+0.095, +0.115] | +0.238 | +0.120 [+0.109, +0.131] |
| debate, simultaneous | all | raw | +0.171 [+0.153, +0.188] | +0.787 | +0.279 [+0.249, +0.306] |
| debate, simultaneous | gemma-3-12b | own map | +0.030 [+0.025, +0.035] | +0.062 | +0.030 [+0.025, +0.035] |
| debate, simultaneous | gemma-3-12b | panel map | +0.062 [+0.051, +0.072] | +0.137 | +0.075 [+0.059, +0.091] |
| debate, simultaneous | gemma-3-12b | raw | +0.117 [+0.093, +0.141] | +0.684 | +0.176 [+0.133, +0.215] |
| debate, simultaneous | gpt-4o-mini | own map | +0.166 [+0.150, +0.181] | +0.372 | +0.166 [+0.150, +0.181] |
| debate, simultaneous | gpt-4o-mini | panel map | +0.182 [+0.164, +0.199] | +0.417 | +0.201 [+0.183, +0.218] |
| debate, simultaneous | gpt-4o-mini | raw | +0.265 [+0.238, +0.290] | +0.834 | +0.374 [+0.334, +0.411] |
| debate, simultaneous | llama-3.1-8b | own map | +0.045 [+0.038, +0.052] | +0.094 | +0.045 [+0.038, +0.052] |
| debate, simultaneous | llama-3.1-8b | panel map | +0.071 [+0.059, +0.081] | +0.161 | +0.085 [+0.073, +0.096] |
| debate, simultaneous | llama-3.1-8b | raw | +0.131 [+0.109, +0.151] | +0.843 | +0.286 [+0.250, +0.322] |

## Prompt optimization (480 held-out items, gpt-5-mini)

`own map`: fitted on these items, so it learns the item selection. `panel map`: fitted on the panel's unselected items and applied here.

| protocol | stage | judge | treatment | ASD, Brier | ASD, log | gain over no argument, Brier |
|---|---|---|---|---|---|---|
| consultancy | after search | all | own map | +0.012 [+0.007, +0.016] | +0.024 | +0.012 [+0.007, +0.016] |
| consultancy | after search | all | panel map | +0.019 [+0.009, +0.028] | +0.041 | +0.035 [+0.025, +0.045] |
| consultancy | after search | all | raw | +0.007 [-0.015, +0.028] | -0.059 | +0.116 [+0.082, +0.150] |
| consultancy | after search | gemma-3-12b | own map | +0.010 [+0.003, +0.017] | +0.020 | +0.010 [+0.003, +0.017] |
| consultancy | after search | gemma-3-12b | panel map | +0.028 [+0.009, +0.047] | +0.064 | +0.042 [+0.022, +0.066] |
| consultancy | after search | gemma-3-12b | raw | +0.044 [+0.009, +0.079] | +0.218 | +0.105 [+0.051, +0.162] |
| consultancy | after search | gpt-4o-mini | own map | +0.025 [+0.014, +0.036] | +0.051 | +0.025 [+0.014, +0.036] |
| consultancy | after search | gpt-4o-mini | panel map | +0.043 [+0.026, +0.059] | +0.088 | +0.062 [+0.045, +0.079] |
| consultancy | after search | gpt-4o-mini | raw | +0.099 [+0.066, +0.130] | +0.269 | +0.207 [+0.152, +0.261] |
| consultancy | after search | llama-3.1-8b | own map | +0.000 [+0.000, +0.000] | +0.000 | +0.000 [+0.000, +0.000] |
| consultancy | after search | llama-3.1-8b | panel map | -0.015 [-0.019, -0.011] | -0.030 | -0.000 [-0.005, +0.004] |
| consultancy | after search | llama-3.1-8b | raw | -0.122 [-0.155, -0.088] | -0.665 | +0.037 [-0.011, +0.083] |
| consultancy | before search | all | own map | +0.033 [+0.025, +0.041] | +0.071 | +0.033 [+0.025, +0.041] |
| consultancy | before search | all | panel map | +0.046 [+0.034, +0.057] | +0.102 | +0.062 [+0.050, +0.073] |
| consultancy | before search | all | raw | +0.080 [+0.056, +0.105] | +0.424 | +0.189 [+0.155, +0.222] |
| consultancy | before search | gemma-3-12b | own map | +0.072 [+0.053, +0.091] | +0.159 | +0.072 [+0.053, +0.091] |
| consultancy | before search | gemma-3-12b | panel map | +0.089 [+0.066, +0.113] | +0.202 | +0.104 [+0.077, +0.129] |
| consultancy | before search | gemma-3-12b | raw | +0.151 [+0.109, +0.195] | +0.958 | +0.212 [+0.154, +0.271] |
| consultancy | before search | gpt-4o-mini | own map | +0.027 [+0.015, +0.038] | +0.055 | +0.027 [+0.015, +0.038] |
| consultancy | before search | gpt-4o-mini | panel map | +0.048 [+0.029, +0.065] | +0.103 | +0.067 [+0.049, +0.084] |
| consultancy | before search | gpt-4o-mini | raw | +0.090 [+0.055, +0.125] | +0.359 | +0.199 [+0.146, +0.252] |
| consultancy | before search | llama-3.1-8b | own map | -0.000 [-0.001, +0.000] | -0.001 | -0.000 [-0.001, +0.000] |
| consultancy | before search | llama-3.1-8b | panel map | +0.000 [-0.004, +0.005] | +0.000 | +0.015 [+0.010, +0.019] |
| consultancy | before search | llama-3.1-8b | raw | -0.002 [-0.042, +0.038] | -0.046 | +0.157 [+0.108, +0.205] |
| debate, sequential | after search | all | own map | +0.051 [+0.041, +0.061] | +0.108 | +0.051 [+0.041, +0.061] |
| debate, sequential | after search | all | panel map | +0.078 [+0.062, +0.093] | +0.174 | +0.094 [+0.077, +0.111] |
| debate, sequential | after search | all | raw | +0.132 [+0.103, +0.157] | +0.322 | +0.241 [+0.199, +0.283] |
| debate, sequential | after search | gemma-3-12b | own map | +0.024 [+0.014, +0.034] | +0.049 | +0.024 [+0.014, +0.034] |
| debate, sequential | after search | gemma-3-12b | panel map | +0.050 [+0.029, +0.070] | +0.103 | +0.064 [+0.037, +0.091] |
| debate, sequential | after search | gemma-3-12b | raw | +0.101 [+0.056, +0.145] | +0.024 | +0.162 [+0.096, +0.230] |
| debate, sequential | after search | gpt-4o-mini | own map | +0.081 [+0.061, +0.100] | +0.171 | +0.081 [+0.061, +0.100] |
| debate, sequential | after search | gpt-4o-mini | panel map | +0.127 [+0.096, +0.157] | +0.296 | +0.147 [+0.117, +0.177] |
| debate, sequential | after search | gpt-4o-mini | raw | +0.184 [+0.138, +0.229] | +0.620 | +0.293 [+0.232, +0.353] |
| debate, sequential | after search | llama-3.1-8b | own map | +0.049 [+0.034, +0.064] | +0.103 | +0.049 [+0.034, +0.064] |
| debate, sequential | after search | llama-3.1-8b | panel map | +0.057 [+0.040, +0.074] | +0.122 | +0.072 [+0.054, +0.089] |
| debate, sequential | after search | llama-3.1-8b | raw | +0.110 [+0.074, +0.145] | +0.323 | +0.268 [+0.216, +0.319] |
| debate, sequential | before search | all | own map | +0.065 [+0.054, +0.076] | +0.141 | +0.065 [+0.054, +0.076] |
| debate, sequential | before search | all | panel map | +0.093 [+0.076, +0.109] | +0.215 | +0.109 [+0.091, +0.126] |
| debate, sequential | before search | all | raw | +0.152 [+0.123, +0.181] | +0.765 | +0.262 [+0.218, +0.302] |
| debate, sequential | before search | gemma-3-12b | own map | +0.010 [+0.004, +0.017] | +0.021 | +0.010 [+0.004, +0.017] |
| debate, sequential | before search | gemma-3-12b | panel map | +0.040 [+0.017, +0.062] | +0.089 | +0.054 [+0.026, +0.081] |
| debate, sequential | before search | gemma-3-12b | raw | +0.072 [+0.021, +0.120] | +0.542 | +0.133 [+0.062, +0.197] |
| debate, sequential | before search | gpt-4o-mini | own map | +0.125 [+0.101, +0.147] | +0.274 | +0.125 [+0.101, +0.147] |
| debate, sequential | before search | gpt-4o-mini | panel map | +0.158 [+0.127, +0.185] | +0.370 | +0.177 [+0.148, +0.204] |
| debate, sequential | before search | gpt-4o-mini | raw | +0.224 [+0.179, +0.267] | +0.769 | +0.333 [+0.273, +0.390] |
| debate, sequential | before search | llama-3.1-8b | own map | +0.061 [+0.045, +0.077] | +0.129 | +0.061 [+0.045, +0.077] |
| debate, sequential | before search | llama-3.1-8b | panel map | +0.082 [+0.062, +0.103] | +0.185 | +0.097 [+0.076, +0.118] |
| debate, sequential | before search | llama-3.1-8b | raw | +0.161 [+0.120, +0.202] | +0.982 | +0.319 [+0.264, +0.375] |
| debate, simultaneous | after search | all | own map | +0.127 [+0.115, +0.139] | +0.302 | +0.127 [+0.115, +0.139] |
| debate, simultaneous | after search | all | panel map | +0.113 [+0.098, +0.127] | +0.248 | +0.129 [+0.113, +0.144] |
| debate, simultaneous | after search | all | raw | +0.177 [+0.150, +0.203] | +0.558 | +0.286 [+0.244, +0.327] |
| debate, simultaneous | after search | gemma-3-12b | own map | +0.040 [+0.026, +0.054] | +0.081 | +0.040 [+0.026, +0.054] |
| debate, simultaneous | after search | gemma-3-12b | panel map | +0.058 [+0.037, +0.080] | +0.120 | +0.072 [+0.046, +0.099] |
| debate, simultaneous | after search | gemma-3-12b | raw | +0.111 [+0.063, +0.159] | -0.055 | +0.171 [+0.105, +0.240] |
| debate, simultaneous | after search | gpt-4o-mini | own map | +0.339 [+0.306, +0.368] | +0.817 | +0.339 [+0.306, +0.368] |
| debate, simultaneous | after search | gpt-4o-mini | panel map | +0.247 [+0.224, +0.269] | +0.541 | +0.266 [+0.243, +0.289] |
| debate, simultaneous | after search | gpt-4o-mini | raw | +0.375 [+0.340, +0.408] | +0.947 | +0.484 [+0.426, +0.542] |
| debate, simultaneous | after search | llama-3.1-8b | own map | +0.004 [-0.001, +0.009] | +0.008 | +0.004 [-0.001, +0.009] |
| debate, simultaneous | after search | llama-3.1-8b | panel map | +0.033 [+0.010, +0.058] | +0.084 | +0.048 [+0.024, +0.072] |
| debate, simultaneous | after search | llama-3.1-8b | raw | +0.045 [+0.001, +0.091] | +0.783 | +0.204 [+0.145, +0.265] |
| debate, simultaneous | before search | all | own map | +0.107 [+0.093, +0.120] | +0.243 | +0.107 [+0.093, +0.120] |
| debate, simultaneous | before search | all | panel map | +0.121 [+0.105, +0.138] | +0.274 | +0.137 [+0.119, +0.155] |
| debate, simultaneous | before search | all | raw | +0.204 [+0.174, +0.234] | +0.828 | +0.313 [+0.268, +0.356] |
| debate, simultaneous | before search | gemma-3-12b | own map | +0.031 [+0.020, +0.042] | +0.064 | +0.031 [+0.020, +0.042] |
| debate, simultaneous | before search | gemma-3-12b | panel map | +0.066 [+0.041, +0.089] | +0.145 | +0.080 [+0.051, +0.107] |
| debate, simultaneous | before search | gemma-3-12b | raw | +0.127 [+0.077, +0.178] | +0.683 | +0.188 [+0.118, +0.258] |
| debate, simultaneous | before search | gpt-4o-mini | own map | +0.230 [+0.199, +0.261] | +0.539 | +0.230 [+0.199, +0.261] |
| debate, simultaneous | before search | gpt-4o-mini | panel map | +0.217 [+0.188, +0.245] | +0.498 | +0.236 [+0.207, +0.265] |
| debate, simultaneous | before search | gpt-4o-mini | raw | +0.317 [+0.273, +0.359] | +0.986 | +0.425 [+0.363, +0.487] |
| debate, simultaneous | before search | llama-3.1-8b | own map | +0.060 [+0.043, +0.076] | +0.125 | +0.060 [+0.043, +0.076] |
| debate, simultaneous | before search | llama-3.1-8b | panel map | +0.082 [+0.059, +0.104] | +0.181 | +0.096 [+0.074, +0.119] |
| debate, simultaneous | before search | llama-3.1-8b | raw | +0.167 [+0.123, +0.210] | +0.816 | +0.325 [+0.266, +0.385] |

## Self-play (390 held-out items, gpt-5-mini; mean of 3 judge samples)

`own map`: fitted on these items, so it learns the item selection. `panel map`: fitted on the panel's unselected items and applied here.

| protocol | stage | judge | treatment | ASD, Brier | ASD, log | gain over no argument, Brier |
|---|---|---|---|---|---|---|
| debate, sequential | equilibrium | all | own map | +0.009 [+0.005, +0.013] | +0.018 | +0.009 [+0.005, +0.013] |
| debate, sequential | equilibrium | all | panel map | -0.003 [-0.019, +0.014] | -0.005 | +0.014 [-0.001, +0.031] |
| debate, sequential | equilibrium | all | raw | +0.005 [-0.027, +0.038] | +0.045 | +0.121 [+0.083, +0.162] |
| debate, sequential | equilibrium | gemma-3-12b | own map | +0.026 [+0.015, +0.038] | +0.054 | +0.026 [+0.015, +0.038] |
| debate, sequential | equilibrium | gemma-3-12b | panel map | +0.039 [+0.023, +0.055] | +0.082 | +0.051 [+0.027, +0.074] |
| debate, sequential | equilibrium | gemma-3-12b | raw | +0.094 [+0.052, +0.136] | +0.294 | +0.146 [+0.080, +0.211] |
| debate, sequential | equilibrium | gpt-4o-mini | own map | +0.000 [+0.000, +0.000] | +0.000 | +0.000 [+0.000, +0.000] |
| debate, sequential | equilibrium | gpt-4o-mini | panel map | -0.048 [-0.086, -0.010] | -0.102 | -0.023 [-0.057, +0.013] |
| debate, sequential | equilibrium | gpt-4o-mini | raw | -0.078 [-0.135, -0.019] | -0.185 | +0.062 [-0.001, +0.129] |
| debate, sequential | equilibrium | llama-3.1-8b | own map | -0.001 [-0.003, +0.002] | -0.001 | -0.001 [-0.003, +0.002] |
| debate, sequential | equilibrium | llama-3.1-8b | panel map | +0.002 [-0.013, +0.018] | +0.005 | +0.015 [-0.000, +0.032] |
| debate, sequential | equilibrium | llama-3.1-8b | raw | -0.001 [-0.040, +0.041] | +0.027 | +0.157 [+0.102, +0.214] |
| debate, sequential | unprompted | all | own map | +0.077 [+0.057, +0.097] | +0.171 | +0.077 [+0.057, +0.097] |
| debate, sequential | unprompted | all | panel map | +0.079 [+0.058, +0.100] | +0.175 | +0.096 [+0.074, +0.118] |
| debate, sequential | unprompted | all | raw | +0.142 [+0.101, +0.182] | +0.463 | +0.258 [+0.208, +0.310] |
| debate, sequential | unprompted | gemma-3-12b | own map | +0.030 [+0.013, +0.047] | +0.062 | +0.030 [+0.013, +0.047] |
| debate, sequential | unprompted | gemma-3-12b | panel map | +0.049 [+0.022, +0.076] | +0.106 | +0.061 [+0.028, +0.094] |
| debate, sequential | unprompted | gemma-3-12b | raw | +0.105 [+0.043, +0.167] | +0.465 | +0.157 [+0.074, +0.241] |
| debate, sequential | unprompted | gpt-4o-mini | own map | +0.136 [+0.101, +0.170] | +0.307 | +0.136 [+0.101, +0.170] |
| debate, sequential | unprompted | gpt-4o-mini | panel map | +0.139 [+0.103, +0.173] | +0.314 | +0.164 [+0.131, +0.198] |
| debate, sequential | unprompted | gpt-4o-mini | raw | +0.205 [+0.150, +0.256] | +0.579 | +0.345 [+0.277, +0.414] |
| debate, sequential | unprompted | llama-3.1-8b | own map | +0.065 [+0.041, +0.091] | +0.145 | +0.065 [+0.041, +0.091] |
| debate, sequential | unprompted | llama-3.1-8b | panel map | +0.050 [+0.032, +0.068] | +0.105 | +0.063 [+0.045, +0.083] |
| debate, sequential | unprompted | llama-3.1-8b | raw | +0.116 [+0.073, +0.162] | +0.346 | +0.274 [+0.212, +0.336] |
| debate, simultaneous | equilibrium | all | own map | +0.092 [+0.075, +0.109] | +0.204 | +0.092 [+0.075, +0.109] |
| debate, simultaneous | equilibrium | all | panel map | +0.086 [+0.070, +0.101] | +0.187 | +0.103 [+0.088, +0.118] |
| debate, simultaneous | equilibrium | all | raw | +0.161 [+0.128, +0.191] | +0.482 | +0.277 [+0.237, +0.317] |
| debate, simultaneous | equilibrium | gemma-3-12b | own map | +0.065 [+0.040, +0.090] | +0.138 | +0.065 [+0.040, +0.090] |
| debate, simultaneous | equilibrium | gemma-3-12b | panel map | +0.071 [+0.045, +0.096] | +0.152 | +0.083 [+0.053, +0.113] |
| debate, simultaneous | equilibrium | gemma-3-12b | raw | +0.161 [+0.099, +0.222] | +0.555 | +0.213 [+0.135, +0.292] |
| debate, simultaneous | equilibrium | gpt-4o-mini | own map | +0.156 [+0.127, +0.186] | +0.355 | +0.156 [+0.127, +0.186] |
| debate, simultaneous | equilibrium | gpt-4o-mini | panel map | +0.143 [+0.117, +0.170] | +0.318 | +0.169 [+0.144, +0.194] |
| debate, simultaneous | equilibrium | gpt-4o-mini | raw | +0.217 [+0.175, +0.258] | +0.580 | +0.357 [+0.302, +0.412] |
| debate, simultaneous | equilibrium | llama-3.1-8b | own map | +0.055 [+0.040, +0.069] | +0.120 | +0.055 [+0.040, +0.069] |
| debate, simultaneous | equilibrium | llama-3.1-8b | panel map | +0.043 [+0.032, +0.055] | +0.091 | +0.057 [+0.045, +0.068] |
| debate, simultaneous | equilibrium | llama-3.1-8b | raw | +0.104 [+0.075, +0.131] | +0.311 | +0.261 [+0.213, +0.306] |
| debate, simultaneous | unprompted | all | own map | +0.107 [+0.086, +0.127] | +0.252 | +0.107 [+0.086, +0.127] |
| debate, simultaneous | unprompted | all | panel map | +0.092 [+0.073, +0.111] | +0.203 | +0.109 [+0.090, +0.130] |
| debate, simultaneous | unprompted | all | raw | +0.163 [+0.127, +0.200] | +0.537 | +0.280 [+0.234, +0.329] |
| debate, simultaneous | unprompted | gemma-3-12b | own map | +0.036 [+0.017, +0.057] | +0.074 | +0.036 [+0.017, +0.057] |
| debate, simultaneous | unprompted | gemma-3-12b | panel map | +0.054 [+0.027, +0.082] | +0.116 | +0.066 [+0.033, +0.100] |
| debate, simultaneous | unprompted | gemma-3-12b | raw | +0.118 [+0.056, +0.181] | +0.473 | +0.170 [+0.091, +0.254] |
| debate, simultaneous | unprompted | gpt-4o-mini | own map | +0.209 [+0.168, +0.249] | +0.510 | +0.209 [+0.168, +0.249] |
| debate, simultaneous | unprompted | gpt-4o-mini | panel map | +0.171 [+0.138, +0.203] | +0.387 | +0.196 [+0.163, +0.229] |
| debate, simultaneous | unprompted | gpt-4o-mini | raw | +0.253 [+0.203, +0.302] | +0.748 | +0.393 [+0.326, +0.463] |
| debate, simultaneous | unprompted | llama-3.1-8b | own map | +0.075 [+0.050, +0.102] | +0.172 | +0.075 [+0.050, +0.102] |
| debate, simultaneous | unprompted | llama-3.1-8b | panel map | +0.051 [+0.033, +0.068] | +0.107 | +0.064 [+0.047, +0.082] |
| debate, simultaneous | unprompted | llama-3.1-8b | raw | +0.118 [+0.076, +0.161] | +0.389 | +0.276 [+0.216, +0.332] |

## The panel split by gap selection (weak judges, fresh verdicts)

| items | protocol | n | judge alone | with the protocol | gain [95% CI] |
|---|---|---|---|---|---|
| gap-selected | debate, simultaneous | 95 | -0.092 | +0.235 | +0.327 [+0.243, +0.413] |
| gap-selected | consultancy (one argument) | 95 | -0.092 | +0.075 | +0.167 [+0.091, +0.239] |
| not selected | debate, simultaneous | 205 | +0.232 | +0.244 | +0.012 [-0.044, +0.067] |
| not selected | consultancy (one argument) | 205 | +0.232 | +0.210 | -0.022 [-0.071, +0.028] |
