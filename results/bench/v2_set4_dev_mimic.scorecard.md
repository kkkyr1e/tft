### tft-bench-v2 · mimic · set4 · dev pool

config d350c3ef563004e4, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator ff4db16ea7fa577e7875bc7a41ab6dcf0a655c81; harness 000930a70c50e1a363526ae65fae537135e7195b

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 4.40 ±0.26 (n=312, SD 2.30) |
| Top-4 rate | 54% [48%, 59%] |
| Win rate | 12% [9%, 17%] |
| Placement histogram 1..8 | 39 39 51 39 34 30 42 38 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.84 ±0.65 | 68% [48%, 83%] | 8% [2%, 25%] |
| 1 | 68 | 4.21 ±0.53 | 63% [51%, 74%] | 10% [5%, 20%] |
| 2 | 106 | 4.33 ±0.46 | 53% [43%, 62%] | 18% [12%, 26%] |
| 3 | 64 | 4.97 ±0.58 | 44% [32%, 56%] | 6% [2%, 15%] |
| 4 | 36 | 4.19 ±0.79 | 53% [37%, 68%] | 19% [10%, 35%] |
| 5 | 11 | 4.91 ±1.31 | 36% [15%, 65%] | 0% [0%, 26%] |
| 6 | 2 | 5.00 ±3.92 | 50% [9%, 91%] | 0% [0%, 66%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 5% [3%, 8%] |
| banked_without_leveling | 90% [86%, 93%] |
| items_on_non_carries | 53% [47%, 58%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 14 / 5 (5) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=168):
- final comps: 8 distinct, entropy 2.85 of 3.00 bits (effective 7.2 comps): divine 38, spirit 26, enlightened 26, elderwood 22, moonlight 22, cultist 17, mage 10, fortune 7
- responsiveness to item at the signal round: MI 0.266 bits (Miller-Madow 0.072; shuffled 0.299, p=0.777)
- responsiveness to chosen at the signal round: MI 0.770 bits (Miller-Madow 0.556; shuffled 0.610, p=0.003)
- responsiveness to contested at the signal round: MI 0.173 bits (Miller-Madow 0.023; shuffled 0.220, p=0.903)
- no comparison agent: variety is descriptive only

Decision-bank regret: not run
