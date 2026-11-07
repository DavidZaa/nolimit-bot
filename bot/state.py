"""Our own view of the game: the types every strategy component is written against.

The real starter kit arrives on event day with its own Agent interface. Rather
than guess at it everywhere, strategy code only ever sees these types, and
adapter.py translates the kit's objects into them. Swapping kits then touches
one file instead of all of them.

Cards are ints: rank * 4 + suit, rank 0..12 meaning 2..A, suit 0..3 meaning
c, d, h, s. That keeps rank (c >> 2) and suit (c & 3) one instruction away,
which matters inside a 1 ms decision budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3

FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"

# Forced bets show up in the action log too, so trackers can tell a blind from a call.
POST_SB = "sb"
POST_BB = "bb"
POST_ANTE = "ante"


def card_from_str(s: str) -> int:
    if len(s) != 2 or s[0] not in RANK_CHARS or s[1] not in SUIT_CHARS:
        raise ValueError(f"not a card: {s!r}")
    return RANK_CHARS.index(s[0]) * 4 + SUIT_CHARS.index(s[1])


def card_to_str(c: int) -> str:
    return RANK_CHARS[c >> 2] + SUIT_CHARS[c & 3]


def cards_from_str(s: str) -> tuple[int, ...]:
    s = s.replace(" ", "")
    return tuple(card_from_str(s[i : i + 2]) for i in range(0, len(s), 2))


class Action(NamedTuple):
    kind: str
    # For RAISE: the total this player's bet becomes on this street ("raise to").
    # Opening bets are raises from zero. Ignored for the other kinds.
    amount: int = 0


class LegalActions(NamedTuple):
    can_fold: bool  # only when there is something to call
    can_check: bool
    can_call: bool
    call_amount: int  # chips added by calling; may be less than the bet if all-in
    can_raise: bool
    min_raise_to: int  # equals max_raise_to when the only raise is a short all-in
    max_raise_to: int


class ActionRecord(NamedTuple):
    seat: int
    street: int
    kind: str  # FOLD/CHECK/CALL/RAISE or a forced post (POST_SB/POST_BB/POST_ANTE)
    amount: int  # street bet total after the action (chips posted, for antes)


@dataclass(frozen=True, slots=True)
class GameRules:
    """Everything about the game a decision depends on, explicit and serializable.

    Built by bot.rules (from the SDK config in the tournament, from the
    TableConfig in the simulator) and validated there. Nothing in bot/ reads
    the variant off the number of hole cards: it reads these fields.
    """

    variant: str  # the SDK's variant ID
    # Cards
    ranks: tuple[int, ...]  # ranks in the deck, ascending (0 = deuce .. 12 = ace)
    suits: tuple[str, ...]  # always c, d, h, s: cards are rank * 4 + suit
    hole_cards: int
    hole_cards_used: int | None  # exactly this many from the hand (Omaha: 2); None = best five of all
    preflop_board: int  # board cards face up before preflop betting (Courchevel: 1)
    board_schedule: tuple[int, ...]  # cards dealt as each later street opens (flop, turn, river)
    category_order: tuple[int, ...]  # hand categories (bot.rules constants), weakest first
    ace_low_straight: bool  # the ace also plays below the lowest four ranks (A-2-3-4-5; short deck A-6-7-8-9)
    betting: str  # "no_limit" or "pot_limit"
    # Tournament (from each table's config)
    payouts: tuple[float, ...]  # points by finishing place, one per seat, 1st first
    starting_stack: int
    blind_schedule: tuple[tuple[int, int, int], ...]  # (small, big, ante) per level; the last repeats
    hands_per_level: int
    hand_cap: int
    seats: int

    @property
    def pot_limit(self) -> bool:
        return self.betting == "pot_limit"

    @property
    def board_size(self) -> int:
        return self.preflop_board + sum(self.board_schedule)

    def deck(self) -> list[int]:
        return [r * 4 + s for r in self.ranks for s in range(len(self.suits))]


@dataclass(frozen=True, slots=True)
class GameState:
    hand_number: int  # 1-based, counts hands played at this table
    seat: int  # whose decision this is
    num_seats: int
    button: int
    sb_seat: int
    bb_seat: int
    small_blind: int
    big_blind: int
    ante: int
    street: int
    hole: tuple[int, ...]
    board: tuple[int, ...]
    stacks: tuple[int, ...]  # chips behind, per seat; 0 for busted or all-in seats
    bets: tuple[int, ...]  # chips put in on the current street, per seat
    pot: int  # everything committed this hand, current street bets included
    in_hand: tuple[bool, ...]  # dealt in and not folded
    players_left: int  # seats that still have chips at the table
    legal: LegalActions
    actions: tuple[ActionRecord, ...]  # this hand so far, forced posts included
    rules: GameRules | None = None


@dataclass(frozen=True, slots=True)
class HandResult:
    hand_number: int
    stacks_after: tuple[int, ...]
    shown: dict[int, tuple[int, ...]]  # seat -> hole cards, only for showdown hands
    winners: tuple[int, ...]
    # The whole hand, forced posts included. PLACEHOLDER(kit): a kit may not
    # report this; the tracker falls back to what it saw in GameState.actions.
    actions: tuple[ActionRecord, ...] = ()
    board: tuple[int, ...] = ()  # the full board, when the hand got that far


def fallback_action(legal: LegalActions) -> Action:
    """The same thing the tournament does on an error or timeout."""
    if legal.can_check:
        return Action(CHECK)
    return Action(FOLD)
