### tft-bench-v1 · stance+hold · set4 · dev pool

config 3bc9bb108cc1c077, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator 2ba01d5a9c25a315e665dd5cafd7ab6ecde2f1a7; harness 1354633d72b2972969ceabdf8e4a93f8d8d7c515

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 3.83 ±0.24 (n=312, SD 2.16) |
| Top-4 rate | 63% [58%, 69%] |
| Win rate | 18% [14%, 23%] |
| Placement histogram 1..8 | 56 47 51 44 32 38 24 20 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.20 ±0.60 | 88% [70%, 96%] | 12% [4%, 30%] |
| 1 | 68 | 3.60 ±0.49 | 60% [48%, 71%] | 24% [15%, 35%] |
| 2 | 106 | 3.94 ±0.43 | 62% [53%, 71%] | 16% [10%, 24%] |
| 3 | 64 | 4.03 ±0.59 | 58% [46%, 69%] | 19% [11%, 30%] |
| 4 | 36 | 3.92 ±0.61 | 69% [53%, 82%] | 14% [6%, 29%] |
| 5 | 11 | 4.09 ±1.60 | 55% [28%, 79%] | 27% [10%, 57%] |
| 6 | 2 | 4.00 ±3.92 | 50% [9%, 91%] | 0% [0%, 66%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 28% [23%, 33%] |
| banked_without_leveling | 28% [23%, 33%] |
| items_on_non_carries | 71% [66%, 76%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 5 / 353 (5) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=198):
- final comps: 8 distinct, entropy 2.80 of 3.00 bits (effective 7.0 comps): spirit 48, divine 42, enlightened 27, moonlight 22, elderwood 21, cultist 17, mage 14, fortune 7
- responsiveness to item at the signal round: MI 0.265 bits (Miller-Madow 0.097; shuffled 0.257, p=0.420)
- responsiveness to chosen at the signal round: MI 0.742 bits (Miller-Madow 0.538; shuffled 0.536, p=0.001)
- responsiveness to contested at the signal round: MI 0.258 bits (Miller-Madow 0.127; shuffled 0.202, p=0.072)
- hero - comparison placement -0.04, 95% upper end +0.24 <= margin 0.25: variety may count as a plus; but not more varied than the comparison

Decision-bank regret: not run
