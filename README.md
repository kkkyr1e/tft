# 云顶调优

云顶之弈调优项目：agent 的评测与调优框架，跑在开源的 S4 模拟器 [TFTMuZeroAgent](https://github.com/silverlight6/TFTMuZeroAgent) 上。

整体方针见 [docs/PLAN.md](docs/PLAN.md)。当前状态是 M0：环境可跑、种子可复现、有两条基线；被测的 LLM agent 还没有实现。

## 安装

```bash
./scripts/setup_sim.sh        # 拉取固定提交的模拟器到 third_party/，并安装依赖
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
python -m pytest tests        # 或者 python tests/test_stats.py
```

需要 Python 3.10 以上。

## 跑一组对局

```bash
# 4 个规则 bot + 4 个随机 bot，28 局
python scripts/run_lobby.py --lobby rule:4,random:4 --games 28 --seed 1000 --workers 2 --out results/rule4_random4.json

# 被测座位记为 hero：1 个规则 bot 对 7 个规则 bot，再换成 20% 动作随机的版本，用同一批种子
python scripts/run_lobby.py --lobby hero=rule:1,rule:7    --games 24 --seed 2000 --out results/hero_rule.json
python scripts/run_lobby.py --lobby hero=noisy20:1,rule:7 --games 24 --seed 2000 --out results/hero_noisy20.json

# 比较两次运行
python scripts/compare.py results/hero_rule.json results/hero_noisy20.json --policy hero
```

可用的策略：`random`、`rule`（模拟器自带的规则 bot）、`noisyN`（有 N% 动作随机的规则 bot）、`mimic` 和它的经济变体 `fast8`、`rolldown8`、`hp50`、`fast8roll`、`fodder2`（计划加执行器，见下）、`llm`（由大模型出计划）。写成 `别名=策略` 可以让某个座位以自己的名字出现在统计里。

## 计划与执行器

计划座位每回合出一份计划，由执行器（改过的规则 bot）展开成原子动作，见 `tfteval/planner.py`。计划的基本字段是 `comp`、`level_to`、`roll_floor`、`carry`。下面这些字段是给策略层用的，不写就和原来完全一样（`tests/test_regression.py` 用改动前录下的动作序列逐个座位核对）：

| 字段 | 含义 |
|---|---|
| `comp` 换成别的 | 换阵容：卖掉备战席上不在新阵容里、又不成对的棋子，把备战席上新阵容的棋子换上场（不弱于被换下的才换） |
| `level_by` `{"level": 8, "by": "4-1"}` | 在该回合前升到 N 级。经验尽量晚买：截止前每回合最多排 10 次，其余在截止回合买，动作不够时抢在规则 bot 前面 |
| `spend` `{"to": 10, "by": "4-3"}` | 在截止回合前把金币 D 到 G，按剩余回合均摊；也可以写 `"rounds": K` |
| `fodder` `true` | 垫子阵容：上最弱的棋子（不足时买 1 费，不跨利息档、不凑三连），强棋子留在备战席，不上装备；改回 `false` 的那回合把最强的换回场上 |
| `survival` `N` | 预计"还能输几把"不超过 N 时，忽略经济字段和垫子，D 到 0，差两次经验以内就升级 |

垫子阵容的冒烟对比（`results/smoke_fodder2.json`，`python scripts/smoke_executor.py --hero fodder2 --games 32 --seed 7300`）：同一批 32 个种子、同一个轮换座位，一次用 `fodder2`（2-1 到 2-6 上垫子），一次用 `mimic`，其余 7 个座位是规则 bot。

| 2-1～2-6 的 5 场对战 | fodder2 | mimic | 同种子配对差 |
|---|---|---|---|
| 输的场数 | 4.28（每局至少输 3 场，一半的局 5 场全输） | 2.53 | +1.75 ±0.51 |
| 掉血 | 34.6 | 18.1 | +16.5 ±4.8 |
| 这几场带来的金币（连胜连败金 + 赢一场 1 金） | 7.6（6.9 + 0.7） | 6.6（4.1 + 2.5） | +1.0 ±1.5 |
| 3-1 开始时：金币 / 棋子价值 / 血量 | 39.9 / 21.5 / 65.4 | 46.7 / 15.8 / 81.9 | |
| 4-1 开始时：金币 / 血量 | 45.1 / 49.5 | 38.7 / 56.0 | +6.4 / −6.4 |
| 最终名次 | 3.28 | 4.00 | −0.72 ±1.11（不显著） |

读法：垫子阵容做得出来，也确实能稳定地输；但在这个模拟器里，多拿的连败金几乎被少拿的赢场金抵掉，第 2 阶段等于用约 16 血换约 1 金。3-1 时金币少，是因为钱压在棋子上（2-7 时 10.8 张对 6.9 张：垫子期间规则 bot 卖多余棋子换利息的动作被拦下，垫子本身也买了约 2 张），金币加棋子价值两边基本相同（61.4 对 62.5）。到 4-1 差距收窄为多 6 金、少 6 血，原因还没查。

`describe()` 给计划者的局面只含公开信息：自己的全部状态；每个对手的场上棋子（星级、装备、天选）、血量、等级、连胜连败、利息档；下回合可能碰到的对手集合（来自 `player.opponent_options`）；各阵容的棋子在对手场上有几张。另有阶段标签（2-1 这种）、离下一次选秀/野怪/换阶段还有几回合、本阶段每输一场的平均扣血和"还能输几把"，见 `tfteval/stages.py`、`tfteval/public.py`。

## 目前的结果

`results/` 里是下面三次运行的原始数据。

| 运行 | 结果 |
|---|---|
| 4 规则 bot + 4 随机 bot，28 局 | 规则 bot 平均名次 2.76 ±0.11，前四率 84%；随机 bot 6.24，前四率 16% |
| 1 规则 bot 对 7 规则 bot，24 局 | 3.79 ±0.90 |
| 1 个 noisy20 对 7 规则 bot，同样的 24 个种子 | 4.08 ±0.90；与上一行的差为 −0.29 ±1.27，同种子相关系数 0.00 |

读法：环境能把规则 bot 和随机 bot 分开，但两个水平接近的版本在 24 局里完全分不开，按种子配对也没有帮助。要把差距定到 ±0.3，每个版本约需 430 局。所以日常调优不能靠整局，要靠决策题库，见方针第 3、6 节。

## 关于复现

同一个种子只有在固定 `PYTHONHASHSEED` 时才会重放出同一局；不固定的话，每换一个进程结果都不同（实测，原因应是模拟器内部依赖哈希顺序）。`scripts/run_lobby.py` 会自动固定为 0；自己写脚本调用 `play_game` 时需要手动设置，结果里的 `reproducible` 字段会标明这一点。

## 对模拟器的修正

模拟器固定在一个提交上，不改 `third_party/` 里的文件；修正写在 `tfteval/simfixes.py`，由 `play_game` 在运行时装上，结果里的 `sim_fixes` 字段记录当局用了哪些修正（空列表表示原版模拟器）。设 `TFT_SIM_FIXES=0` 可以关掉，用来重放修正之前的运行。修正前后的结果不能混在一起比。

| 修正 | 原版的问题 | 修正后 |
|---|---|---|
| `carousel_order` | 选秀顺序的循环只把"血量不高于当前队首"的玩家插到队首，其余玩家拿不到选秀单位。实测 3 局：第一次选秀 8 人里只有 1 人拿到，之后每次 1～5 人 | 每个活着的玩家都拿一次。第一次选秀所有人同时放出，之后从血量最低起两人一组放出，同血量随机。每人仍拿费用最高的单位，与原版相同 |

已知但没有修的问题（对所有座位一样，改了会改变基线，先记下）：

| 问题 | 位置 | 影响 |
|---|---|---|
| 商店里只要有一格空着，购买掩码就全部关闭（`shop_empty` 实际判断的是"有空格"） | `game/player.py` `shop_empty`，`encoding/token/action.py` | 按掩码行动的策略（包括规则 bot）每次刷新只能买 1 张卡 |
| 规则 bot 把备战席棋子换上场的检查永远不触发（`棋子对象 in 名字列表`） | `generators/default_agent.py` 两处 `in BASE_CHAMPION_LIST` | 规则 bot 只在有空位时上人，阵容选择只影响买和卖，不影响上场的棋子 |

## 目录

| 路径 | 内容 |
|---|---|
| `tfteval/runner.py` | 打一整局并给出名次 |
| `tfteval/policies.py` | 座位策略：随机、规则 bot、带噪声的规则 bot、计划座位 |
| `tfteval/planner.py` | 计划者（规则经济、大模型）、计划编译、执行器 |
| `tfteval/stages.py` | 回合序号与阶段标签、赛程、扣血表、"还能输几把" |
| `tfteval/public.py` | 公开观察：对手能被看到的部分 |
| `tfteval/stats.py` | 平均名次与区间、配对差、所需局数 |
| `scripts/` | 批量对局、比较两次运行、安装模拟器、测扣血表（`measure_damage.py`）、执行器冒烟对比（`smoke_executor.py`） |
| `results/` | 原始对局结果 |
| `docs/PLAN.md` | 方针 |

## 许可

模拟器为 Apache-2.0 许可，由 `scripts/setup_sim.sh` 单独拉取，不包含在本仓库内。
