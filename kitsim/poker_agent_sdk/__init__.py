"""Stand-in for the starter kit's poker_agent_sdk, from the organizers' published description.

The real SDK only arrives at kickoff. This mirrors what the kit page says a
bot sees, so adapter.py can be written and tested now: an Agent base class
with act(state, memory) -> (Action, bytes), Action constructors, legal
actions with min/max amounts, and a state carrying events, seats, blinds and
the table config. On kit day, diff this against the real SDK first: every
name here is a PLACEHOLDER(kit).

Local only. It never ships: pack.sh copies agent.py and bot/ alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0

    @classmethod
    def fold(cls) -> "Action":
        return cls("fold")

    @classmethod
    def check(cls) -> "Action":
        return cls("check")

    @classmethod
    def call(cls) -> "Action":
        return cls("call")

    @classmethod
    def raise_to(cls, total: int) -> "Action":
        return cls("raise", int(total))


@dataclass(frozen=True)
class LegalAction:
    kind: str
    min_amount: int = 0
    max_amount: int = 0


@dataclass(frozen=True)
class Event:
    kind: str
    seat: int | None = None
    amount: int | None = None
    cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class Seat:
    seat: int
    stack: int
    bet: int
    status: str  # active, folded, all_in or busted
    shown_cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    table_id: str = "local"
    seat: int = 0
    variant: str = "nlhe"
    players: int = 8
    starting_stack: int = 2000
    # The published fields only: the variant is an ID, its card rules are in RULES.md.
    # PLACEHOLDER(kit): the blinds schedule's real shape; we accept (small, big[, ante])
    # sequences, dicts or objects with small/big/ante.
    blinds: tuple = ((10, 20, 0),)  # the schedule: (small, big, ante) per level
    hands_per_level: int = 20
    hand_cap: int = 250
    payouts: tuple[float, ...] = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    decision_ms: int = 1
    time_bank_ms: int = 100
    memory_bytes: int = 65536


@dataclass(frozen=True)
class State:
    seat: int
    cards: tuple[str, ...]
    board: tuple[str, ...]
    pot: int
    to_call: int
    legal_actions: tuple[LegalAction, ...]
    seats: tuple[Seat, ...]
    button: int
    blinds: tuple[int, ...]  # current (small, big[, ante])
    hand_number: int
    players_remaining: int
    config: Config
    events: tuple[Event, ...] = ()
    history: tuple[Event, ...] = ()
    phase: str = "betting"
    blind_level: int = 1
    hands_until_next_level: int = 0
    bust_order: tuple[int, ...] = ()
    variant: dict = field(default_factory=dict)
    time_bank_ms: int = 100

    def can(self, kind: str) -> LegalAction | None:
        for a in self.legal_actions:
            if a.kind == kind:
                return a
        return None


class Agent:
    def act(self, state: State, memory: bytes):
        raise NotImplementedError
