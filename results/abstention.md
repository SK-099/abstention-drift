# Abstention results

## S2_mtragun

Overall: abstain F1 0.443 | recall 0.464 | precision 0.425 | false-abstention 0.184 (n=429)

| Bucket | n unans. | Abstain recall | n ans. | False-abstention |
| --- | --- | --- | --- | --- |
| early | 13 | 0.538 | 90 | 0.156 |
| mid | 28 | 0.393 | 135 | 0.222 |
| late | 56 | 0.482 | 107 | 0.159 |

**Abstention drift (recall slope / turn):** -0.0030 [95% CI -0.0461, +0.0425], 97 conversations
**False-abstention slope / turn:** -0.0056 [-0.0216, +0.0120]
**Controlled logit (domain + follow-up type):** odds ratio per turn 0.995

## S5-noselfcheck_mtragun

Overall: abstain F1 0.465 | recall 0.443 | precision 0.489 | false-abstention 0.136 (n=429)

| Bucket | n unans. | Abstain recall | n ans. | False-abstention |
| --- | --- | --- | --- | --- |
| early | 13 | 0.462 | 90 | 0.111 |
| mid | 28 | 0.357 | 135 | 0.148 |
| late | 56 | 0.482 | 107 | 0.140 |

**Abstention drift (recall slope / turn):** +0.0100 [95% CI -0.0326, +0.0543], 97 conversations
**False-abstention slope / turn:** +0.0019 [-0.0129, +0.0188]
**Controlled logit (domain + follow-up type):** odds ratio per turn 1.055

## S5-selective_mtragun

Overall: abstain F1 0.511 | recall 0.577 | precision 0.459 | false-abstention 0.199 (n=429)

| Bucket | n unans. | Abstain recall | n ans. | False-abstention |
| --- | --- | --- | --- | --- |
| early | 13 | 0.538 | 90 | 0.167 |
| mid | 28 | 0.500 | 135 | 0.230 |
| late | 56 | 0.625 | 107 | 0.187 |

**Abstention drift (recall slope / turn):** +0.0204 [95% CI -0.0262, +0.0683], 97 conversations
**False-abstention slope / turn:** -0.0008 [-0.0185, +0.0186]
**Controlled logit (domain + follow-up type):** odds ratio per turn 1.096
