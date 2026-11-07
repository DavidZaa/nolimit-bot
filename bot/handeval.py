"""Complete hand evaluation for the packaged bot, under explicit rules. Standard library only.

A hand's value is one int that compares directly (higher wins, equal ties):

    strength of its category under rules.category_order * 13**5 + five tiebreak ranks

the same encoding as the simulator's reference evaluator (engine/handeval.py),
so the two can be checked against each other hand for hand.

Which five cards make the hand follows the rules, never the hole-card count:
rules.hole_cards_used None means the best five of all cards (hold'em, short
deck), k means exactly k from the hand and 5 - k from the board (Omaha: 2).
The best hand is the best over every legal five, which keeps any category
order exact (short deck's flush over full house and trips over straight
included). Each five is scored by a table lookup on its sorted ranks, built
once per deck, ace rule and category order, plus a flush check, so a full
seven-card hold'em hand costs 21 lookups and a PLO river 60.

Straights come from the deck's own ranks: five consecutive ranks of the deck,
and with ace_low_straight the ace below the four lowest (A-2-3-4-5, or
A-6-7-8-9 in short deck, which ranks just under 6-7-8-9-T).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, combinations_with_replacement

from bot.rules import (
    FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH, TRIPS, TWO_PAIR,
)

BASE = 13 ** 5  # room for five base-13 tiebreak ranks under each category


def _value(tiebreak) -> int:
    v = 0
    for i in range(5):
        v = v * 13 + (tiebreak[i] if i < len(tiebreak) else 0)
    return v


@lru_cache(maxsize=16)
def _tables(ranks: tuple[int, ...], ace_low: bool, order: tuple[int, ...]):
    """(score of a non-flush five by its ranks sorted high to low, score of a flush five likewise).

    Every five-rank pattern the deck allows, scored once: category strength * BASE + tiebreaks.
    """
    strength = [0] * 9
    for s, cat in enumerate(order):
        strength[cat] = s
    straights = {frozenset(ranks[i:i + 5]): ranks[i + 4] for i in range(len(ranks) - 4)}
    if ace_low and len(ranks) >= 5 and ranks[-1] == 12:
        straights[frozenset((ranks[-1], *ranks[:4]))] = ranks[3]
    plain: dict[tuple, int] = {}
    suited: dict[tuple, int] = {}
    for combo in combinations_with_replacement(reversed(ranks), 5):  # each sorted high to low
        counts: dict[int, int] = {}
        for r in combo:
            counts[r] = counts.get(r, 0) + 1
        if max(counts.values()) > 4:
            continue
        groups = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
        shape = sorted(counts.values(), reverse=True)
        high = straights.get(frozenset(combo)) if len(counts) == 5 else None
        if high is not None:
            plain[combo] = strength[STRAIGHT] * BASE + _value((high,))
            suited[combo] = strength[STRAIGHT_FLUSH] * BASE + _value((high,))
            continue
        if shape[0] == 4:
            cat, tb = QUADS, groups
        elif shape[:2] == [3, 2]:
            cat, tb = FULL_HOUSE, groups
        elif shape[0] == 3:
            cat, tb = TRIPS, groups
        elif shape[:2] == [2, 2]:
            cat, tb = TWO_PAIR, groups
        elif shape[0] == 2:
            cat, tb = PAIR, groups
        else:
            cat, tb = HIGH_CARD, combo
            suited[combo] = strength[FLUSH] * BASE + _value(combo)
        plain[combo] = strength[cat] * BASE + _value(tb)
    return plain, suited


def _tables_for(rules):
    return _tables(tuple(rules.ranks), rules.ace_low_straight, tuple(rules.category_order))


def warm(rules) -> None:
    """Build these rules' lookup tables now (about 10 ms) instead of on the first postflop decision."""
    _tables_for(rules)


def score5(cards, rules) -> int:
    """Value of exactly five cards."""
    plain, suited = _tables_for(rules)
    key = tuple(sorted((c >> 2 for c in cards), reverse=True))
    s = cards[0] & 3
    try:
        return suited[key] if all(c & 3 == s for c in cards) else plain[key]
    except KeyError:
        raise ValueError(f"not five cards from this deck ({rules.variant}): {cards}") from None


def hand_value(hole, board, rules) -> int:
    """Value of the best legal hand from `hole` and `board` under `rules`.

    Raises ValueError when the cards can't make a legal five yet (too few, or
    not enough board for an exact-usage rule).
    """
    if rules is None:
        raise ValueError("explicit rules required")
    plain, suited = _tables_for(rules)
    used = rules.hole_cards_used
    if used is None:
        cards = (*hole, *board)
        if len(cards) < 5:
            raise ValueError(f"{len(cards)} cards can't make a five-card hand")
        fives = combinations(cards, 5)
    else:
        if len(hole) < used or len(board) < 5 - used:
            raise ValueError(f"need {used} hole cards and {5 - used} board cards, have {len(hole)} and {len(board)}")
        boards = list(combinations(board, 5 - used))
        fives = ((*h, *b) for h in combinations(hole, used) for b in boards)
    best = -1
    try:
        for five in fives:
            a, b, c, d, e = five
            key = tuple(sorted((a >> 2, b >> 2, c >> 2, d >> 2, e >> 2), reverse=True))
            s = a & 3
            v = suited[key] if b & 3 == s and c & 3 == s and d & 3 == s and e & 3 == s else plain[key]
            if v > best:
                best = v
    except KeyError:
        raise ValueError(f"cards outside this deck ({rules.variant}): {hole} {board}") from None
    return best


def category(value: int, rules) -> int:
    """The hand category (bot.rules constants) of a value from hand_value or score5."""
    return rules.category_order[value // BASE]


def tiebreak(value: int) -> tuple[int, ...]:
    """The five tiebreak ranks of a value, most significant first (0s pad short ones)."""
    v = value % BASE
    out = []
    for _ in range(5):
        v, r = divmod(v, 13)
        out.append(r)
    return tuple(reversed(out))
