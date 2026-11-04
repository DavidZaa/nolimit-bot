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


def made_hand(hole, board) -> int:
    """Bucket a postflop hand. A pair that's only on the board counts as nothing."""
    if len(hole) + len(board) < 5:
        return NOTHING
    if len(hole) == 4:
        level = max(_level((*h, *b)) for h in combinations(hole, 2) for b in combinations(board, 3))
    else:
        level = _level((*hole, *board))
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
