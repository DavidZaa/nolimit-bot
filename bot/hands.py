"""Postflop hand reading for the strategy, on top of the complete evaluator in bot.handeval.

hand_strength() gives the whole picture: the hand's exact value (comparable
across hands), its category under the variant's order, and the coarse bucket
the push/fold strategy was written and measured against:

    NOTHING < WEAK_PAIR < TOP_PAIR < TWO_PAIR_PLUS < MONSTER (trips or better)

The buckets read a pair the way a player does: a pair that's only on the
board is nothing, a pocket pair over the board or a pair with the board's
top card is top pair, anything lower is weak. Strategy that needs to tell a
straight from a flush or a full house reads hand_strength().category or
.value instead; the frozen v5 rules deliberately treat all of them as one.

Every function takes the table's rules (bot.state.GameRules); nothing here
reads the variant off the number of hole cards.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import NamedTuple

from bot.handeval import category, hand_value, tiebreak
from bot.rules import FLUSH, FULL_HOUSE, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH, TRIPS, TWO_PAIR, RulesError

# Made-hand buckets, weakest first.
NOTHING, WEAK_PAIR, TOP_PAIR, TWO_PAIR_PLUS, MONSTER = range(5)
_MONSTERS = frozenset((TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH))


class HandStrength(NamedTuple):
    value: int  # bot.handeval value: compare hands with it
    category: int  # bot.rules category constant
    bucket: int  # NOTHING .. MONSTER


def _bucket(cat: int, value: int, hole, board) -> int:
    if cat in _MONSTERS:
        return MONSTER
    if cat == TWO_PAIR:
        return TWO_PAIR_PLUS
    if cat != PAIR:
        return NOTHING
    pair = tiebreak(value)[0]
    mine = [c >> 2 for c in hole].count(pair)
    if not mine:
        return NOTHING  # the pair is the board's
    top = max(c >> 2 for c in board)
    if mine >= 2:  # a pocket pair
        return TOP_PAIR if pair > top else WEAK_PAIR
    return TOP_PAIR if pair == top else WEAK_PAIR


def hand_strength(hole, board, rules) -> HandStrength | None:
    """The best legal hand now, or None before there is one (preflop, or too little board)."""
    try:
        value = hand_value(hole, board, rules)
    except ValueError:
        return None
    cat = category(value, rules)
    return HandStrength(value, cat, _bucket(cat, value, hole, board))


def made_hand(hole, board, rules) -> int:
    """The strategy's bucket for the best legal hand (NOTHING before there is one)."""
    if rules is None:
        raise RulesError("explicit rules required")
    s = hand_strength(hole, board, rules)
    return NOTHING if s is None else s.bucket


@lru_cache(maxsize=16)
def _straights(ranks: tuple[int, ...], ace_low: bool) -> tuple[frozenset, ...]:
    out = [frozenset(ranks[i:i + 5]) for i in range(len(ranks) - 4)]
    if ace_low and len(ranks) >= 5 and ranks[-1] == 12:
        out.append(frozenset((ranks[-1], *ranks[:4])))
    return tuple(out)


def straight_outs(hole, board, rules) -> frozenset[int]:
    """Ranks that, as the next board card, would complete a straight this hand can use."""
    hole_ranks = {c >> 2 for c in hole}
    board_ranks = {c >> 2 for c in board}
    used = rules.hole_cards_used
    outs = set()
    for straight in _straights(tuple(rules.ranks), rules.ace_low_straight):
        if used is None:
            missing = straight - hole_ranks - board_ranks
            if len(missing) == 1 and straight & hole_ranks:
                outs |= missing
            continue
        for mine in combinations(sorted(straight & hole_ranks), used):
            missing = straight - set(mine) - board_ranks
            if len(missing) == 1:
                outs |= missing
    return frozenset(outs)


def flush_draw(hole, board, rules) -> bool:
    """One more suited board card would give this hand a flush it can use."""
    used = rules.hole_cards_used
    for suit in {c & 3 for c in hole}:
        mine = sum(1 for c in hole if c & 3 == suit)
        theirs = sum(1 for c in board if c & 3 == suit)
        if used is None:
            if mine + theirs == 4:
                return True
        elif mine >= used and theirs == 4 - used:
            return True
    return False


def strong_draw(hole, board, rules) -> bool:
    """A flush draw or a straight draw with two or more completing ranks (open-ender, double gutter)."""
    if len(board) >= rules.board_size:
        return False
    return flush_draw(hole, board, rules) or len(straight_outs(hole, board, rules)) >= 2
