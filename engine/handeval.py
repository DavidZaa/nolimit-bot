"""Reference hand evaluator for the simulator.

Deliberately simple: score every five-card combination and keep the best.
It is slower than a lookup-table evaluator, but it is easy to trust, handles
any variant in rules.py, and doubles as the oracle the bot's fast evaluator
gets tested against.

Scores are ints that compare directly: higher wins, equal ties.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

from engine.rules import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    Variant,
)

_BASE = 13**5  # room for five base-13 tiebreak ranks under each category


@lru_cache(maxsize=None)
def _straights(ranks: tuple[int, ...]) -> dict[frozenset[int], int]:
    """Every five-rank straight in this deck, mapped to its high card.

    The wheel is the ace plus the four lowest ranks, so short deck's A-6-7-8-9
    falls out naturally (nine-high, below 6-7-8-9-T).
    """
    r = sorted(ranks)
    out = {frozenset(r[i : i + 5]): r[i + 4] for i in range(len(r) - 4)}
    if len(r) >= 5:
        out[frozenset([r[-1], *r[:4]])] = r[3]
    return out


@lru_cache(maxsize=None)
def _strength(order: tuple[int, ...]) -> tuple[int, ...]:
    strength = [0] * len(order)
    for s, cat in enumerate(order):
        strength[cat] = s
    return tuple(strength)


def evaluate5(cards, variant: Variant) -> int:
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    flush = len({c & 3 for c in cards}) == 1

    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Ranks ordered by how many we hold, then by rank: AAKKQ -> [A, K, Q].
    groups = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = sorted(counts.values(), reverse=True)

    straight_high = _straights(variant.ranks).get(frozenset(ranks)) if len(counts) == 5 else None

    if straight_high is not None:
        cat, tiebreak = (STRAIGHT_FLUSH if flush else STRAIGHT), [straight_high]
    elif shape[0] == 4:
        cat, tiebreak = QUADS, groups
    elif shape[:2] == [3, 2]:
        cat, tiebreak = FULL_HOUSE, groups
    elif flush:
        cat, tiebreak = FLUSH, ranks
    elif shape[0] == 3:
        cat, tiebreak = TRIPS, groups
    elif shape[:2] == [2, 2]:
        cat, tiebreak = TWO_PAIR, groups
    elif shape[0] == 2:
        cat, tiebreak = PAIR, groups
    else:
        cat, tiebreak = HIGH_CARD, ranks

    value = 0
    for i in range(5):
        value = value * 13 + (tiebreak[i] if i < len(tiebreak) else 0)
    return _strength(variant.category_order)[cat] * _BASE + value


def best_hand(hole, board, variant: Variant) -> int:
    k = variant.hole_cards_used
    if k is None:
        return max(evaluate5(combo, variant) for combo in combinations((*hole, *board), 5))
    return max(
        evaluate5((*h, *b), variant)
        for h in combinations(hole, k)
        for b in combinations(board, 5 - k)
    )


def category(score: int, variant: Variant) -> int:
    return variant.category_order[score // _BASE]
