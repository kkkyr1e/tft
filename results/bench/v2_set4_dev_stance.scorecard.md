### tft-bench-v2 · stance · set4 · dev pool

config d350c3ef563004e4, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator ff4db16ea7fa577e7875bc7a41ab6dcf0a655c81; harness 000930a70c50e1a363526ae65fae537135e7195b

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 4.09 ±0.24 (n=312, SD 2.19) |
| Top-4 rate | 56% [51%, 62%] |
| Win rate | 16% [12%, 21%] |
| Placement histogram 1..8 | 50 44 38 44 43 37 36 20 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.96 ±0.75 | 64% [45%, 80%] | 12% [4%, 30%] |
| 1 | 68 | 4.13 ±0.50 | 56% [44%, 67%] | 13% [7%, 23%] |
| 2 | 106 | 4.07 ±0.46 | 56% [46%, 65%] | 21% [14%, 29%] |
| 3 | 64 | 3.83 ±0.53 | 62% [50%, 73%] | 20% [12%, 32%] |
| 4 | 36 | 4.69 ±0.67 | 44% [30%, 60%] | 6% [2%, 18%] |
| 5 | 11 | 3.91 ±1.31 | 55% [28%, 79%] | 9% [2%, 38%] |
| 6 | 2 | 4.50 ±0.98 | 50% [9%, 91%] | 0% [0%, 66%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 3% [2%, 6%] |
| banked_without_leveling | 27% [22%, 32%] |
| items_on_non_carries | 73% [68%, 78%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 12 / 10 (9) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=176):
- final comps: 8 distinct, entropy 2.83 of 3.00 bits (effective 7.1 comps): divine 36, spirit 36, cultist 26, elderwood 25, moonlight 19, enlightened 18, mage 10, fortune 6
- responsiveness to item at the signal round: MI 0.367 bits (Miller-Madow 0.187; shuffled 0.290, p=0.046)
- responsiveness to chosen at the signal round: MI 0.921 bits (Miller-Madow 0.716; shuffled 0.582, p=0.001)
- responsiveness to contested at the signal round: MI 0.240 bits (Miller-Madow 0.100; shuffled 0.215, p=0.238)
- no comparison agent: variety is descriptive only

Decision-bank regret: not run
