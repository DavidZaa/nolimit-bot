"""Postflop hand reading for the submitted bot: stdlib only, no tables to load.

Decisions only need to know roughly how strong the made hand is, so this
buckets it (nothing / weak pair / top pair / two pair / trips or better)
straight from rank and suit counts instead of scoring five-card combinations.
Hold'em hands use the best five of all the cards; four-card hands are read
as Omaha, exactly two from the hand.

PLACEHOLDER(kit): GameState doesn't carry the variant, so the hole-card
count decides Omaha, and A-2-3-4-5 is taken as the low straight.
"""

from __future__ import annotations

from itertools import combinations

# Made-hand buckets, weakest first.
NOTHING, WEAK_PAIR, TOP_PAIR, TWO_PAIR_PLUS, MONSTER = range(5)

_HIGH, _PAIR, _TWO_PAIR, _STRONG = range(4)  # trips, straights and flushes are all _STRONG


def _level(cards) -> int:
    """How far the best five of `cards` gets, from counts alone."""
    suits = [0, 0, 0, 0]
    counts = [0] * 13
    for c in cards:
        suits[c & 3] += 1
        counts[c >> 2] += 1
    if max(suits) >= 5 or max(counts) >= 3:
        return _STRONG
    run = 1 if counts[12] else 0  # the ace also plays low
    for n in counts:
        run = run + 1 if n else 0
        if run >= 5:
            return _STRONG
    pairs = counts.count(2)
    return _TWO_PAIR if pairs >= 2 else _PAIR if pairs else _HIGH


_WHEEL = frozenset((12, 0, 1, 2, 3))


def _omaha_level(hole, board) -> int:
    """Best level over every two-from-the-hand, three-from-the-board five.

    Each five is classified by how many distinct ranks it has (5: straight,
    flush or nothing; 4: a pair; 3: trips or two pair; fewer: full house or
    quads), which is far cheaper than counting, and the search stops at the
    first strong five.
    """
    pairs = [((a >> 2, b >> 2), a & 3 if a & 3 == b & 3 else -1) for a, b in combinations(hole, 2)]
    best = _HIGH
    for x, y, z in combinations(board, 3):
        b_ranks = (x >> 2, y >> 2, z >> 2)
        b_suit = x & 3 if x & 3 == y & 3 == z & 3 else -2
        for h_ranks, h_suit in pairs:
            ranks = {*b_ranks, *h_ranks}
            distinct = len(ranks)
            if distinct == 5:
                if h_suit == b_suit or max(ranks) - min(ranks) == 4 or ranks == _WHEEL:
                    return _STRONG
            elif distinct == 4:
                if best == _HIGH:
                    best = _PAIR
            elif distinct == 3:
                five = b_ranks + h_ranks
                if any(five.count(r) == 3 for r in ranks):
                    return _STRONG
                best = _TWO_PAIR
            else:
                return _STRONG
    return best


def made_hand(hole, board) -> int:
    """Bucket a postflop hand. A pair that's only on the board counts as nothing."""
    if len(hole) + len(board) < 5:
        return NOTHING
    level = _omaha_level(hole, board) if len(hole) == 4 else _level((*hole, *board))
    if level == _STRONG:
        return MONSTER
    if level == _TWO_PAIR:
        return TWO_PAIR_PLUS
    if level == _HIGH:
        return NOTHING
    board_ranks = {c >> 2 for c in board}
    top = max(board_ranks)
    hole_ranks = [c >> 2 for c in hole]
    pockets = [r for r in set(hole_ranks) if hole_ranks.count(r) >= 2]
    if pockets:
        return TOP_PAIR if max(pockets) > top else WEAK_PAIR
    paired = [r for r in set(hole_ranks) if r in board_ranks]
    if not paired:
        return NOTHING
    return TOP_PAIR if max(paired) == top else WEAK_PAIR
