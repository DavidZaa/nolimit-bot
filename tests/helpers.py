"""Small bots and builders shared by the engine tests."""

from __future__ import annotations

import random

from bot.state import CALL, CHECK, FOLD, RAISE, Action, fallback_action


class Scripted:
    """Plays a fixed list of actions, then checks/folds. Keeps every state it saw."""

    def __init__(self, *actions: Action):
        self.queue = list(actions)
        self.seen = []

    def act(self, state):
        self.seen.append(state)
        return self.queue.pop(0) if self.queue else fallback_action(state.legal)


class RandomLegal:
    """Uniformly random legal action; raises pick min, max, or something between.

    Folds are down-weighted so hands actually reach later streets and showdowns.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, state):
        legal = state.legal
        rng = self.rng
        options = []
        if legal.can_fold and rng.random() < 0.35:
            options.append(Action(FOLD))
        if legal.can_check:
            options.append(Action(CHECK))
        if legal.can_call:
            options.append(Action(CALL))
        if legal.can_raise and rng.random() < 0.5:
            lo, hi = legal.min_raise_to, legal.max_raise_to
            options.append(Action(RAISE, rng.choice((lo, hi, rng.randint(lo, hi)))))
        if not options:
            return fallback_action(legal)
        return rng.choice(options)


class Exploding:
    def act(self, state):
        raise RuntimeError("bot crashed")


class ReturnsJunk:
    def __init__(self, value):
        self.value = value

    def act(self, state):
        return self.value
