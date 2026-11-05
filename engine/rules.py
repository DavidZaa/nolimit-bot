"""Tournament rules as data, so a new variant or blind schedule is a config change.

Anything we are assuming rather than reading from the official rules is marked
PLACEHOLDER(kit): the starter kit on event day replaces those values.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.icm import PAYOUTS

# Hand categories. The numbers are identities, not strengths: a variant's
# category_order decides which beats which.
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
)
STANDARD_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH)


@dataclass(frozen=True)
class Variant:
    name: str
    ranks: tuple[int, ...] = tuple(range(13))  # which ranks are in the deck (0 = deuce)
    hole_cards: int = 2
    board_schedule: tuple[int, ...] = (3, 1, 1)  # cards dealt on flop, turn, river
    category_order: tuple[int, ...] = STANDARD_ORDER  # weakest first
    hole_cards_used: int | None = None  # None: best five of all cards; 2: Omaha
    pot_limit: bool = False

    def deck(self) -> list[int]:
        return [r * 4 + s for r in self.ranks for s in range(4)]


NLHE = Variant("nlhe")
# Triton-style short deck: 6-A, flush beats full house, trips beat straight.
# PLACEHOLDER(kit): some rule sets (PokerKit's included) keep straight > trips.
SHORT_DECK = Variant(
    "short_deck",
    ranks=tuple(range(4, 13)),
    category_order=(HIGH_CARD, PAIR, TWO_PAIR, STRAIGHT, TRIPS, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH),
)
PLO = Variant("plo", hole_cards=4, hole_cards_used=2, pot_limit=True)
VARIANTS = {v.name: v for v in (NLHE, SHORT_DECK, PLO)}


@dataclass(frozen=True)
class BlindSchedule:
    levels: tuple[tuple[int, int, int], ...]  # (small blind, big blind, ante) per level
    hands_per_level: int = 25

    def level_for(self, hand_number: int) -> tuple[int, int, int]:
        i = (hand_number - 1) // self.hands_per_level
        return self.levels[min(i, len(self.levels) - 1)]


# PLACEHOLDER(kit): only "start 10/20, up every 25 hands" is published. These two
# bracket the plausible range so strategy gets tuned against both.
DOUBLING = BlindSchedule(tuple((10 * 2**i, 20 * 2**i, 0) for i in range(8)))
X1_5 = BlindSchedule(tuple((bb // 2, bb, 0) for bb in (20, 30, 50, 70, 100, 150, 230, 350)))
SCHEDULES = {"doubling": DOUBLING, "x1.5": X1_5}

# Points by finishing place. One setting, in bot/icm.py, so the simulator
# always scores by the same points the bot plays for.
PLACEHOLDER_PAYOUTS = PAYOUTS


@dataclass(frozen=True)
class TableConfig:
    variant: Variant = NLHE
    schedule: BlindSchedule = DOUBLING
    payouts: tuple[float, ...] = PLACEHOLDER_PAYOUTS
    seats: int = 8
    starting_stack: int = 2000
    max_hands: int = 200
    time_limit_ns: int = 1_000_000  # the official 1 ms per decision
    enforce_time: bool = True  # treat slow decisions exactly like the tournament does
