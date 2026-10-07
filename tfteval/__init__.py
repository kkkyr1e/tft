"""tfteval: evaluation harness for Teamfight Tactics agents on the Set 4 simulator."""

from tfteval.policies import NoisyRulePolicy, Policy, RandomPolicy, RuleBotPolicy, make_policy
from tfteval.runner import Game, GameResult, play_game
from tfteval.stats import paired_diff, summarize

__all__ = [
    "Game",
    "GameResult",
    "NoisyRulePolicy",
    "Policy",
    "RandomPolicy",
    "RuleBotPolicy",
    "make_policy",
    "paired_diff",
    "play_game",
    "summarize",
]
