"""Starting-hand ranking for the submitted bot, rebuilt per variant at kickoff.

A starting hand's class is what's left once suits are interchangeable: the
set of ranks it holds in each suit. AsKs and AhKh share a class, AsKh is
another, and the same key covers hold'em (169 classes), short deck and
Omaha (16,432).

gen/hand_ranks.py simulates every class against random hands and writes
bot/hand_tables/<variant id>.py: a percentile per class ("top x% of starting
hands"), the same scale the Chen ranking in bot.baseline uses, stamped with
the card rules it was simulated under.

ranks_for(rules) picks the table: the one named after the table's variant ID,
used only if its stamped card rules equal the table's (deck, hole cards,
Omaha usage, board, category order, ace rule). With no matching table it's
Chen: exact for two-card hands, the best two cards for bigger ones (an
approximation; pack.sh reports which a release uses). Plain hold'em has no
table on purpose: simulated rankings measured no better than Chen there.

Five-card Omaha has too many classes (about 134,000) to simulate at kickoff,
so its table works through sub-hands: a hand is judged by its best
four-card sub-hand on the four-card table, rescaled against sampled
five-card hands so the result is still a share of five-card hands.
"""

from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
from importlib import import_module
from itertools import combinations

from bot.baseline import preflop_percentile as chen_percentile


def hand_class(hole) -> tuple[tuple[int, ...], ...]:
    """Suit-free class of a starting hand: its rank sets per suit, best first."""
    by_suit: list[list[int]] = [[], [], [], []]
    for c in hole:
        by_suit[c & 3].append(c >> 2)
    return tuple(sorted((tuple(sorted(r, reverse=True)) for r in by_suit if r), reverse=True))


class HandRanks:
    def __init__(self, percentiles=None, hole_cards=None, variant="chen",
                 subset=None, quantiles=None):
        """`subset`/`quantiles`: judge `hole_cards`-card hands by their best `subset`-card
        sub-hand in `percentiles`, rescaled by the sorted sub-hand scores of sampled hands."""
        self.percentiles = percentiles or {}
        self.hole_cards = hole_cards
        self.variant = variant
        self.subset = subset
        self.quantiles = quantiles

    @classmethod
    def from_module(cls, t) -> "HandRanks":
        return cls(t.PERCENTILES, t.HOLE_CARDS, t.VARIANT,
                   getattr(t, "SUBSET", None), getattr(t, "QUANTILES", None))

    def percentile(self, hole) -> float:
        """Share of starting hands at least this good: near 0 for the best, near 1 for the worst."""
        if len(hole) == self.hole_cards:
            if self.subset:
                get = self.percentiles.get
                best = min(get(hand_class(sub), 1.0) for sub in combinations(hole, self.subset))
                q = self.quantiles
                return (min(bisect_left(q, best), len(q) - 1) + 0.5) / len(q)
            p = self.percentiles.get(hand_class(hole))
            if p is not None:
                return p
        return chen_percentile(hole)


CHEN = HandRanks()

# The card rules a table depends on (betting structure and payouts don't change equities).
TABLE_RULES = ("ranks", "hole_cards", "hole_cards_used", "preflop_board", "board_schedule",
               "category_order", "ace_low_straight")


def card_fingerprint(rules) -> tuple:
    get = rules.get if isinstance(rules, dict) else lambda k: getattr(rules, k)
    return tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in (get(k) for k in TABLE_RULES))


def _table_module(variant: str):
    if not variant.isidentifier() or variant.startswith("_"):
        return None
    try:
        return import_module(f"bot.hand_tables.{variant}")
    except ImportError:
        return None


@lru_cache(maxsize=32)
def _ranks(variant: str, fingerprint: tuple) -> tuple[HandRanks, str]:
    t = _table_module(variant)
    if t is None:
        return CHEN, f"no table for {variant!r}: Chen ranking"
    if card_fingerprint(getattr(t, "CARD_RULES", {k: None for k in TABLE_RULES})) != fingerprint:
        return CHEN, f"bot/hand_tables/{variant}.py was made for other card rules: Chen ranking"
    return HandRanks.from_module(t), f"bot/hand_tables/{variant}.py"


def ranks_for(rules) -> HandRanks:
    """The starting-hand ranking for these rules (see the module docstring)."""
    return _ranks(rules.variant, card_fingerprint(rules))[0]


def table_status(rules) -> str:
    """Which ranking ranks_for(rules) uses, in words (for release checks)."""
    return _ranks(rules.variant, card_fingerprint(rules))[1]
