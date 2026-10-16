### tft-bench-v1 · stance · set4 · dev pool

config 3bc9bb108cc1c077, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator 2ba01d5a9c25a315e665dd5cafd7ab6ecde2f1a7; harness 1354633d72b2972969ceabdf8e4a93f8d8d7c515

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 3.87 ±0.23 (n=312, SD 2.06) |
| Top-4 rate | 62% [57%, 68%] |
| Win rate | 14% [11%, 19%] |
| Placement histogram 1..8 | 45 52 52 46 42 33 26 16 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.60 ±0.86 | 76% [57%, 89%] | 16% [6%, 35%] |
| 1 | 68 | 3.74 ±0.48 | 63% [51%, 74%] | 18% [10%, 28%] |
| 2 | 106 | 4.04 ±0.37 | 60% [51%, 69%] | 11% [7%, 19%] |
| 3 | 64 | 3.56 ±0.52 | 67% [55%, 77%] | 17% [10%, 28%] |
| 4 | 36 | 4.28 ±0.73 | 53% [37%, 68%] | 11% [4%, 25%] |
| 5 | 11 | 4.18 ±1.21 | 55% [28%, 79%] | 9% [2%, 38%] |
| 6 | 2 | 3.50 ±4.90 | 50% [9%, 91%] | 50% [9%, 91%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 27% [23%, 32%] |
| banked_without_leveling | 36% [30%, 41%] |
| items_on_non_carries | 71% [66%, 76%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 6 / 389 (6) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=195):
- final comps: 8 distinct, entropy 2.84 of 3.00 bits (effective 7.2 comps): spirit 39, divine 33, elderwood 31, cultist 28, enlightened 26, moonlight 20, mage 13, fortune 5
- responsiveness to item at the signal round: MI 0.325 bits (Miller-Madow 0.159; shuffled 0.259, p=0.059)
- responsiveness to chosen at the signal round: MI 0.984 bits (Miller-Madow 0.799; shuffled 0.537, p=0.001)
- responsiveness to contested at the signal round: MI 0.204 bits (Miller-Madow 0.071; shuffled 0.195, p=0.382)
- no comparison agent: variety is descriptive only

Decision-bank regret: not run
