### tft-bench-v2 · stance+hold · set4 · dev pool

config d350c3ef563004e4, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator ff4db16ea7fa577e7875bc7a41ab6dcf0a655c81; harness 000930a70c50e1a363526ae65fae537135e7195b, f177700b5f05ff643ffc31d72e22d74a0923f145

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 4.25 ±0.24 (n=312, SD 2.19) |
| Top-4 rate | 55% [50%, 61%] |
| Win rate | 13% [10%, 17%] |
| Placement histogram 1..8 | 40 42 45 45 41 40 29 30 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.96 ±0.66 | 56% [37%, 73%] | 12% [4%, 30%] |
| 1 | 68 | 3.91 ±0.53 | 66% [54%, 76%] | 16% [9%, 27%] |
| 2 | 106 | 4.28 ±0.43 | 53% [43%, 62%] | 14% [9%, 22%] |
| 3 | 64 | 4.22 ±0.57 | 58% [46%, 69%] | 14% [8%, 25%] |
| 4 | 36 | 4.89 ±0.60 | 39% [25%, 55%] | 0% [0%, 10%] |
| 5 | 11 | 4.36 ±1.38 | 55% [28%, 79%] | 18% [5%, 48%] |
| 6 | 2 | 7.00 ±0.00 | 0% [0%, 66%] | 0% [0%, 66%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 3% [1%, 5%] |
| banked_without_leveling | 22% [18%, 27%] |
| items_on_non_carries | 73% [68%, 78%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 13 / 9 (9) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=172):
- final comps: 8 distinct, entropy 2.85 of 3.00 bits (effective 7.2 comps): spirit 35, divine 29, elderwood 27, cultist 23, moonlight 22, enlightened 21, mage 9, fortune 6
- responsiveness to item at the signal round: MI 0.353 bits (Miller-Madow 0.173; shuffled 0.297, p=0.120)
- responsiveness to chosen at the signal round: MI 0.941 bits (Miller-Madow 0.753; shuffled 0.564, p=0.001)
- responsiveness to contested at the signal round: MI 0.259 bits (Miller-Madow 0.113; shuffled 0.233, p=0.260)
- no comparison agent: variety is descriptive only

Decision-bank regret: not run
