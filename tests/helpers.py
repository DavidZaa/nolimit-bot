"""Small bots and builders shared by the engine tests."""

from __future__ import annotations

import random

from bot.state import RAISE, Action, cards_from_str, fallback_action
from engine.field import FolderBot
from engine.field import RandomBot as RandomLegal  # noqa: F401  (shared with the runner)
from engine.rules import TableConfig
from engine.table import play_hand


class Scripted:
    """Plays a fixed list of actions, then checks/folds. Keeps every state it saw."""

    def __init__(self, *actions: Action):
        self.queue = list(actions)
        self.seen = []

    def act(self, state):
        self.seen.append(state)
        return self.queue.pop(0) if self.queue else fallback_action(state.legal)


class Exploding:
    def act(self, state):
        raise RuntimeError("bot crashed")


class ReturnsJunk:
    def __init__(self, value):
        self.value = value

    def act(self, state):
        return self.value


def decisions(bot, seat, hole, board, stacks=None, others=None, config=None):
    """Play one hand (button 0, blinds 10/20) with `bot` at `seat`; return its (state, action)s.

    Everyone else folds unless given in `others`; they get low filler cards.
    `config` sets the variant and points (default: TableConfig()).
    """
    stacks = stacks or [2000] * 8
    seen = []

    class Spy:
        def act(self, state):
            action = bot.act(state)
            seen.append((state, action))
            return action

    bots = [FolderBot() for _ in range(8)]
    for s, other in (others or {}).items():
        bots[s] = other
    bots[seat] = Spy()
    mine, shared = cards_from_str(hole), cards_from_str(board)
    spare = iter(c for c in range(52) if c not in mine + shared)
    holes = {s: mine if s == seat else (next(spare), next(spare)) for s in range(8) if stacks[s] > 0}
    play_hand(bots, stacks, 0, 1, config or TableConfig(), random.Random(0), holes=holes, board=shared)
    return seen


def first_action(*args, **kw):
    return decisions(*args, **kw)[0][1]


def facing_shove(bot, seat, hole, shover=3, stacks=None, config=None):
    """`bot`'s answer when `shover` moves all-in first. Builds a fresh shover each call."""
    stacks = stacks or [2000] * 8
    state, action = decisions(bot, seat, hole, "Kh7s2dJc4h", stacks=stacks, config=config,
                              others={shover: Scripted(Action(RAISE, stacks[shover]))})[0]
    assert max(state.bets) == stacks[shover]  # it really was facing the shove
    return action
