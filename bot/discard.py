"""Which hole card to throw away when the variant makes us keep two of three. Pure function.

    choose_discard(hole, board, rules=None) -> the card to discard

Every possible keep (three of them) is scored, and the best keep wins:

  1. its made hand's category, under the variant's category order (a set beats
     two pair beats a pair beats nothing), from the complete evaluator;
  2. then draws: flush draw 4, open-ender 3, gutshot 2, plus 1 per overcard to
     the board;
  3. then the exact hand value (higher kickers, a higher pair).

Before there's a flop to read (if the kit ever asks that early), the keep with
the best two-card starting-hand percentile wins. Ties go to the lower card
index so the choice is deterministic. About 30 us per call.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

from bot.baseline import preflop_percentile
from bot.handeval import BASE, hand_value
from bot.hands import flush_draw, straight_outs

FLUSH_DRAW, OPEN_ENDER, GUTSHOT, OVERCARD = 4, 3, 2, 1


@lru_cache(maxsize=1)
def default_rules():
    """The registered discard variant's rules (with placeholder table settings): enough to
    evaluate hands when a caller has no GameRules of its own."""
    from bot.rules import make_rules

    return make_rules("three_card_discard", payouts=(50, 30, 20), starting_stack=2000,
                      blind_schedule=((10, 20, 0),), hands_per_level=20, hand_cap=250, seats=8)


def draw_points(keep, board, rules) -> int:
    if len(board) >= rules.board_size:
        return 0  # nothing left to draw to
    outs = straight_outs(keep, board, rules)
    points = FLUSH_DRAW if flush_draw(keep, board, rules) else 0
    points += OPEN_ENDER if len(outs) >= 2 else GUTSHOT if outs else 0
    top = max((c >> 2 for c in board), default=-1)
    return points + sum(OVERCARD for c in keep if c >> 2 > top)


def keep_score(keep, board, rules) -> tuple:
    """How good a two-card keep is with this board (bigger is better)."""
    if len(board) + len(keep) < 5:
        return (0, 0, -preflop_percentile(keep))
    value = hand_value(keep, board, rules)
    return (value // BASE, draw_points(keep, board, rules), value)


def choose_discard(hole, board, rules=None) -> int:
    """The card to discard from `hole` (normally three cards) so the best keep remains."""
    rules = rules or default_rules()
    hole = tuple(hole)
    keep_size = len(hole) - max(1, rules.discards)
    best = max(combinations(range(len(hole)), keep_size),
               key=lambda idx: (keep_score([hole[i] for i in idx], board, rules), [-i for i in idx]))
    return next(c for i, c in enumerate(hole) if i not in best)
