"""Which kept hole card to show when the variant makes us expose one. Pure function.

    choose_expose(kept, board, rules=None) -> the card to show

Strong made hand (top pair or better): show the card that contributes least to
it, so the table sees as little of our strength as possible. A card contributes
by pairing the board (or the other kept card), by sharing a suit the board
holds two or more of, and by sitting in a five-rank straight window with the
board. Ties show the lower card.

Weak hand or a draw: show the scariest card, to make our bets credible: an
ace, then a card of the board's main suit, then the highest card.
"""

from __future__ import annotations

from collections import Counter

from bot.discard import default_rules
from bot.hands import TOP_PAIR, hand_strength
from bot.hands import _straights as straights


def contribution(card, other, board, rules) -> int:
    rank, suit = card >> 2, card & 3
    board_ranks = Counter(c >> 2 for c in board)
    board_suits = Counter(c & 3 for c in board)
    score = 3 * board_ranks[rank] + (3 if other >> 2 == rank else 0)
    if board_suits[suit] >= 2:
        score += 2
    present = set(board_ranks) | {rank}
    if any(rank in s and len(s & present) >= 4 for s in straights(tuple(rules.ranks), rules.ace_low_straight)):
        score += 1
    return score


def scariest(kept, board) -> int:
    aces = [c for c in kept if c >> 2 == 12]
    if aces:
        return min(aces)
    suits = Counter(c & 3 for c in board)
    main = [s for s, n in suits.items() if n >= 2]
    on_suit = [c for c in kept if c & 3 in main]
    if len(on_suit) == 1:
        return on_suit[0]
    pool = on_suit or kept
    return max(pool, key=lambda c: (c >> 2, -c))


def choose_expose(kept, board, rules=None) -> int:
    rules = rules or default_rules()
    kept = tuple(kept)
    if len(kept) == 1:
        return kept[0]
    strength = hand_strength(kept, board, rules)
    if strength is not None and strength.bucket >= TOP_PAIR:
        a, b = kept[0], kept[1]
        return min(kept, key=lambda c: (contribution(c, b if c == a else a, board, rules), c >> 2, c))
    return scariest(kept, board)
