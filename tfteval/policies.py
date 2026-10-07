"""Seat policies. A policy maps (observation, info, agent, env) to one simulator action.

Every policy owns its randomness so that a game is a function of (seed, lobby) only.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

MASK_SHAPE = (55, 38)  # default full-game action space, flattened by the env


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation, info, agent, env): ...


def _action_class(env):
    from Simulator.encoding.token.action import ActionToken  # imported lazily: needs the simulator on PYTHONPATH

    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "action_class", None) or ActionToken


class RandomPolicy:
    """Uniform over legal actions, drawn from a private generator."""

    name = "random"

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation, info, agent, env):
        mask = np.asarray(observation["action_mask"]).reshape(-1)
        legal = np.flatnonzero(mask > 0)
        if len(legal) == 0:
            raise RuntimeError("no legal action in mask")
        return _action_class(env).action_space_to_action(int(legal[self._rng.integers(len(legal))]))


class RuleBotPolicy:
    """The simulator's hand-written Default_Agent.

    The upstream bot indexes the action mask as a 2-D array while the env emits it flat,
    so the mask is reshaped here. The bot also draws from numpy's global generator when it
    picks a team comp; the runner seeds that generator once per game.
    """

    name = "rule"

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation, info, agent, env):
        from Simulator.utils import decode_action

        player = info["player"]
        mask = np.asarray(observation["action_mask"])
        if mask.ndim == 1 and mask.size == MASK_SHAPE[0] * MASK_SHAPE[1]:
            mask = mask.reshape(MASK_SHAPE)
        command = player.default_policy(info.get("game_round", 1), info.get("shop"), mask)
        return decode_action([command])[0]


class NoisyRulePolicy:
    """Rule bot that plays a uniformly random legal action with probability `epsilon`.

    Gives a ladder of opponents whose strength order is known in advance
    (random < noisy50 < noisy20 < rule), which is what lets the benchmark itself be checked.
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.name = f"noisy{round(epsilon * 100)}"
        self._rule, self._random = RuleBotPolicy(), RandomPolicy()
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._random.reset(seed + 1)

    def act(self, observation, info, agent, env):
        if self._rng.random() < self.epsilon:
            return self._random.act(observation, info, agent, env)
        return self._rule.act(observation, info, agent, env)


_REGISTRY = {"random": RandomPolicy, "rule": RuleBotPolicy}


def make_policy(spec: str) -> Policy:
    """Build a policy from a spec: `random`, `rule`, `noisy20` (rule bot with 20% random
    actions), `mimic` (plan executor with the rule bot's economy) or `llm` (plan executor with a
    model planner, see tfteval/planner.py). Prefix `alias=` to report a seat under its own name, e.g. `hero=rule`."""
    alias, _, kind = spec.rpartition("=")
    from tfteval.planner import VARIANTS, make_plan_policy

    if kind in VARIANTS or kind == "llm":
        policy = make_plan_policy(kind)
    elif kind.startswith("noisy") and kind[5:].isdigit():
        policy = NoisyRulePolicy(int(kind[5:]) / 100)
    elif kind in _REGISTRY:
        policy = _REGISTRY[kind]()
    else:
        raise ValueError(f"unknown policy {kind!r}; choose from {sorted(_REGISTRY)} or noisy<percent>")
    if alias:
        policy.name = alias
    return policy
