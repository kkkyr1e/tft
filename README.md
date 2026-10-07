# 云顶调优

云顶之弈调优项目：agent 的评测与调优框架，跑在开源的 S4 模拟器 [TFTMuZeroAgent](https://github.com/silverlight6/TFTMuZeroAgent) 上，用的是我们修过 bug 的 fork：[kkkyr1e/TFTMuZeroAgent](https://github.com/kkkyr1e/TFTMuZeroAgent) 的 `develop` 分支。

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

模拟器用我们的 fork [kkkyr1e/TFTMuZeroAgent](https://github.com/kkkyr1e/TFTMuZeroAgent)，`scripts/setup_sim.sh` 固定在 `develop` 的一个提交上。bug 修在 fork 里，每个修正是一个单独的分支、带一个单元测试，可以单独给上游提 PR；`develop` 合并了全部修正。清单、出处和测试见 fork 里的 `FORK_NOTES.md`。

| 修正 | 原版的问题 | 修正后 |
|---|---|---|
| 选秀顺序 | 只有血量不高于队首的玩家能拿到单位，第一次选秀 8 人里只有 1 人拿到 | 人人有份；第一次所有人同时，之后从低血起两人一组 |
| 阶段伤害 | 每档伤害提前一个阶段生效，第 2 阶段就有基础伤害 | 按 10.24 版本的 S4 表：第 1～7 阶段基础伤害 0/0/2/3/5/8/15 |
| 6-7 野怪 | 被跳过 | 正常打 |
| 前期收入 | 1-2、1-3 没有收入，金币和经验晚两回合 | 1-2/1-3/1-4/2-1 给 2/2/3/4 金，每回合 2 经验 |
| 匹配 | 加权抽签会抽到不该遇到的对手 | 只在可遇到的对手里按权重抽 |
| 买牌 | 每次刷新只能买一张 | 商店里剩下的都能买 |
| 座位顺序 | 座位放在 set 里，同一个种子在不同进程里是不同的对局 | 固定顺序；实测同一种子在 `PYTHONHASHSEED` 为 0 和 1 时名次完全一致 |
| 规则 bot | Katarina 拼错；备战席换上场的逻辑是死代码 | 修好 |
| 天选价格 | 2～5 费天选比正式游戏便宜 1 金 | 1 星价格的 3 倍 |

另外 fork 里加了一个选项（默认关）：跳过就结束本回合，并可调高每回合的动作上限。

**这次换模拟器以后，之前所有的结果都不能再拿来比**：伤害、收入、买牌、匹配和规则 bot 都变了。结果里的 `sim_commit` 字段记录当局用的是哪个模拟器提交。

`tfteval/simfixes.py` 是换 fork 之前在运行时打的补丁，现在模拟器自带修正，它就什么都不装（`sim_fixes` 为空列表）；只有用原版上游模拟器时才会装上。

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
