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

可用的策略：`random`、`rule`（模拟器自带的规则 bot）、`noisyN`（有 N% 动作随机的规则 bot）、`llm`（占位，未实现）。写成 `别名=策略` 可以让某个座位以自己的名字出现在统计里。

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

## 目录

| 路径 | 内容 |
|---|---|
| `tfteval/runner.py` | 打一整局并给出名次 |
| `tfteval/policies.py` | 座位策略：随机、规则 bot、带噪声的规则 bot、LLM 占位 |
| `tfteval/stats.py` | 平均名次与区间、配对差、所需局数 |
| `scripts/` | 批量对局、比较两次运行、安装模拟器 |
| `results/` | 原始对局结果 |
| `docs/PLAN.md` | 方针 |

## 许可

模拟器为 Apache-2.0 许可，由 `scripts/setup_sim.sh` 单独拉取，不包含在本仓库内。
