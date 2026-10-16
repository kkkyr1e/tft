### tft-bench-v1 · mimic · set4 · dev pool

config 3bc9bb108cc1c077, sim realistic, rules set4; 312/312 games (seeds 30000-30311), 0 unfinished; simulator 2ba01d5a9c25a315e665dd5cafd7ab6ecde2f1a7; harness 1354633d72b2972969ceabdf8e4a93f8d8d7c515

| Strength | Value |
|---|---|
| Mean placement (95% CI) | 4.59 ±0.27 (n=312, SD 2.40) |
| Top-4 rate | 48% [43%, 54%] |
| Win rate | 13% [10%, 17%] |
| Placement histogram 1..8 | 41 42 31 37 39 33 38 51 |
| Policy errors (hero / all seats) | 0 / 0 |

| Stance-family opponents | Games | Mean placement | Top-4 | Win |
|---|---|---|---|---|
| 0 | 25 | 3.60 ±0.80 | 64% [45%, 80%] | 16% [6%, 35%] |
| 1 | 68 | 4.29 ±0.56 | 62% [50%, 72%] | 10% [5%, 20%] |
| 2 | 106 | 4.52 ±0.46 | 44% [35%, 54%] | 18% [12%, 26%] |
| 3 | 64 | 4.77 ±0.58 | 47% [35%, 59%] | 9% [4%, 19%] |
| 4 | 36 | 5.44 ±0.75 | 31% [18%, 47%] | 11% [4%, 25%] |
| 5 | 11 | 5.55 ±1.62 | 36% [15%, 65%] | 9% [2%, 38%] |
| 6 | 2 | 5.00 ±5.88 | 50% [9%, 91%] | 0% [0%, 66%] |

| Rubric v1 (hero, per game) | Rate |
|---|---|
| died_with_gold | 71% [66%, 76%] |
| banked_without_leveling | 96% [93%, 98%] |
| items_on_non_carries | 55% [49%, 60%] |
| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | 0 / 17 / 737 (5) / 0 |

Strategic variety (reported apart from strength; top-4 finishes only, n=151):
- final comps: 8 distinct, entropy 2.84 of 3.00 bits (effective 7.2 comps): divine 29, spirit 27, cultist 25, enlightened 20, moonlight 20, elderwood 18, mage 6, fortune 6
- responsiveness to item at the signal round: MI 0.440 bits (Miller-Madow 0.244; shuffled 0.336, p=0.022)
- responsiveness to chosen at the signal round: MI 0.961 bits (Miller-Madow 0.789; shuffled 0.661, p=0.001)
- responsiveness to contested at the signal round: MI 0.341 bits (Miller-Madow 0.188; shuffled 0.265, p=0.055)
- hero - comparison placement +0.72, 95% upper end +0.99 > margin 0.25: variety is not reported as a plus

Decision-bank regret: not run
