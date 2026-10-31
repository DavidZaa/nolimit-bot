"""Small bots and builders shared by the engine tests."""

from __future__ import annotations

from bot.state import Action, fallback_action
from engine.field import RandomBot as RandomLegal  # noqa: F401  (shared with the runner)


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
