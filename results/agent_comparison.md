# S5-selective vs S2 on mtragun (429 turns)

| Gold label | S2 | S5-selective | Turns |
| --- | --- | --- | --- |
| answerable | abstain | abstain | 4 |
| answerable | abstain | answer | 16 |
| answerable | abstain | clarify | 41 |
| answerable | answer | abstain | 18 |
| answerable | answer | answer | 250 |
| answerable | answer | clarify | 3 |
| unanswerable | abstain | abstain | 1 |
| unanswerable | abstain | answer | 2 |
| unanswerable | abstain | clarify | 42 |
| unanswerable | answer | abstain | 12 |
| unanswerable | answer | answer | 39 |
| unanswerable | answer | clarify | 1 |

## Effect of the agent steps

- Answerable turns S2 refused that the agent now answers (recovered): **16**
- Unanswerable turns S2 refused that the agent now answers (lost abstentions): **2**
- Unanswerable turns S2 answered that the agent now refuses (new catches): **13**
- Answerable turns S2 answered that the agent now refuses (new wrong refusals): **21**
- Turns retried: 117
- Self-check abstentions: {'ans': 21, 'unans': 13} (unans = correct, ans = wrong)
- Clarifying questions: {'unans': 43, 'ans': 44}
