# Per-model refusal thresholds (MTRAG checker dev, 150 answerable / 74 unanswerable)

Abstain if p >= tau, or the model refuses, or (p >= min_p and the self-check says unsupported).
FA = answerable questions refused. Own refusals = FA with no gate and no self-check.

| Model | Thresholds | tau | min_p | F1 | Recall | FA | Own refusals | s / question |
|---|---|---|---|---|---|---|---|---|
| local:gemma3:4b | calibrated | 0.64 | 0.55 | 0.67 | 0.85 | 0.34 | 0.33 | 36.65 |
| local:qwen3:1.7b | calibrated | 0.63 | 0.35 | 0.67 | 0.72 | 0.21 | 0.15 | 32.49 |
| local:granite4:3b | calibrated | 0.61 | 0.45 | 0.64 | 0.88 | 0.43 | 0.38 | 20.11 |
| local:qwen2.5:3b | research | 0.61 | 0.50 | 0.62 | 0.73 | 0.30 | 0.22 | - |
| local:llama3.2:3b | calibrated | 0.45 | 0.35 | 0.62 | 0.89 | 0.48 | 0.38 | 16.65 |
| local:phi4-mini | calibrated | 0.52 | off | 0.60 | 0.92 | 0.57 | 0.54 | 21.43 |
| local:qwen2.5:1.5b | calibrated | 0.61 | 0.35 | 0.56 | 0.49 | 0.12 | 0.02 | 9.67 |
| local:smollm2:1.7b | calibrated | 0.61 | 0.50 | 0.54 | 0.46 | 0.12 | 0.02 | 17.60 |
| local:deepseek-r1:1.5b | calibrated | 0.54 | off | 0.54 | 0.51 | 0.19 | 0.09 | 38.12 |
| local:llama3.2:1b | calibrated | 0.52 | off | 0.53 | 0.72 | 0.48 | 0.38 | 7.08 |
