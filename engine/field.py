"""Sparring partners modelled on what a 60-minute field tends to produce.

These are the easy archetypes. LLM-style "default TAG" bots come later; the
point here is a varied, reproducible field to run the harness against.
"""

from __future__ import annotations

import random

from bot.state import CALL, CHECK, FOLD, RAISE, Action, fallback_action


class FolderBot:
    """A bot that crashes or times out every hand: the tournament checks/folds for it."""

    def act(self, state):
        return fallback_action(state.legal)


class CallingStation:
    def act(self, state):
        legal = state.legal
        return Action(CALL) if legal.can_call else Action(CHECK)


class Maniac:
    """Shoves every decision it can."""

    def act(self, state):
        legal = state.legal
        if legal.can_raise:
            return Action(RAISE, legal.max_raise_to)
        return Action(CALL) if legal.can_call else Action(CHECK)


class RandomBot:
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
