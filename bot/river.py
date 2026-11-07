"""Exact river equity in Bay Area, where every opponent's hidden card is one of a few dozen.

After the reveal an opponent holds one public card and one hidden one. On the
river the hidden card is one of the deck minus everything known: our two cards,
the board, every exposed card (folded players' too) and our own discard, at
most 44. So our equity against that opponent is exact: the share of those
cards we beat, ties counting half. Multiway, the lowest equity over the
opponents still in is what we act on.

eval7 (in the kit's requirements) scores a seven-card hand in under half a
microsecond, so one opponent costs ~20 us; without it our own evaluator does
the same at ~8 us a hand, within the budget heads-up only. The caller passes
the budget it can afford and gets None when the work wouldn't fit.
"""

from __future__ import annotations

from time import perf_counter_ns

from bot.handeval import hand_value
from bot.rules import STANDARD_ORDER
from bot.state import card_to_str

try:
    import eval7 as _eval7

    _CARDS = [_eval7.Card(card_to_str(c)) for c in range(52)]
    _evaluate = _eval7.evaluate
except Exception:  # not installed, or not importable here
    _eval7 = None
    _CARDS = []
    _evaluate = None

EVAL7_NS = 1_000  # about what one seven-card eval7 evaluation costs, with margin
OWN_NS = 12_000  # ... with our evaluator


def standard_rules(rules) -> bool:
    """eval7 knows only standard hold'em scoring: 52 cards, best five of all, usual order."""
    return (tuple(rules.ranks) == tuple(range(13)) and rules.hole_cards_used is None
            and tuple(rules.category_order) == STANDARD_ORDER and rules.ace_low_straight)


def river_equity(hole, board, opponents: list[int], dead, rules, budget_ns: int) -> float | None:
    """Our equity (win + half of ties) against the weakest-case opponent, or None if it
    wouldn't fit `budget_ns` or any opponent's exposed card is unknown (-1)."""
    if not opponents or any(e < 0 for e in opponents):
        return None
    known = set(hole) | set(board) | set(dead) | set(opponents)
    candidates = [c for c in rules.deck() if c not in known]
    fast = _evaluate is not None and standard_rules(rules)
    per_eval = EVAL7_NS if fast else OWN_NS
    if (len(candidates) * len(opponents) + 1) * per_eval > budget_ns:
        return None
    start = perf_counter_ns()
    if fast:
        board_cards = [_CARDS[c] for c in board]
        mine = _evaluate([_CARDS[c] for c in hole] + board_cards)
    else:
        mine = hand_value(hole, board, rules)
    worst = 1.0
    for exposed in opponents:
        score = 0.0
        for hidden in candidates:
            if fast:
                theirs = _evaluate([_CARDS[exposed], _CARDS[hidden]] + board_cards)
            else:
                theirs = hand_value((exposed, hidden), board, rules)
            if mine > theirs:
                score += 1.0
            elif mine == theirs:
                score += 0.5
        worst = min(worst, score / len(candidates))
        if perf_counter_ns() - start > budget_ns:
            return None
    return worst
