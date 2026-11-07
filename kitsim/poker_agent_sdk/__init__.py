"""Stand-in for the starter kit's poker_agent_sdk, for runs and tests without the kit installed.

Our own minimal re-statement of the shapes in the kit's poker_agent_protocol/game.py
(checked against it on 2026-11-06): Action(kind, amount, cards), LegalAction,
Event(kind, seat, amount=0, cards), Seat, BlindLevel(small_blind, big_blind, ante)
for state.blinds and every config.blinds level, the table config and the state.
tests/test_real_sdk.py runs the adapter on the real types whenever the kit is installed.

Local only. It never ships: pack.sh copies agent.py and bot/ alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0
    cards: tuple[int, ...] = ()  # whole numbers for a variant's own kind (Bay Area: select indexes)

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
    amount: int = 0
    cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    ante: int = 0


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
    blinds: tuple = (BlindLevel(10, 20, 0),)  # the schedule; the last level repeats
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
    blinds: BlindLevel  # the current level
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
