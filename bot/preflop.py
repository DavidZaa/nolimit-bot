"""Starting-hand ranking for the submitted bot, rebuilt per variant at kickoff.

A starting hand's class is what's left once suits are interchangeable: the
set of ranks it holds in each suit. AsKs and AhKh share a class, AsKh is
another, and the same key covers hold'em (169 classes), short deck and
Omaha (16,432).

gen/hand_ranks.py simulates every class against random hands and writes
bot/hand_ranks_table.py: a percentile per class ("top x% of starting
hands"), the same scale the Chen ranking in bot.baseline uses. The bot uses
that table when it has one for its hole-card count, and Chen otherwise.
"""

from __future__ import annotations

from bot.baseline import preflop_percentile as chen_percentile


def hand_class(hole) -> tuple[tuple[int, ...], ...]:
    """Suit-free class of a starting hand: its rank sets per suit, best first."""
    by_suit: list[list[int]] = [[], [], [], []]
    for c in hole:
        by_suit[c & 3].append(c >> 2)
    return tuple(sorted((tuple(sorted(r, reverse=True)) for r in by_suit if r), reverse=True))


class HandRanks:
    def __init__(self, percentiles=None, hole_cards=None, variant="chen"):
        self.percentiles = percentiles or {}
        self.hole_cards = hole_cards
        self.variant = variant

    @classmethod
    def load(cls) -> "HandRanks":
        """The generated table if there is one, else Chen everywhere."""
        try:
            from bot import hand_ranks_table as t
        except ImportError:
            return cls()
        return cls(t.PERCENTILES, t.HOLE_CARDS, t.VARIANT)

    def percentile(self, hole) -> float:
        """Share of starting hands at least this good: near 0 for the best, near 1 for the worst."""
        if len(hole) == self.hole_cards:
            p = self.percentiles.get(hand_class(hole))
            if p is not None:
                return p
        return chen_percentile(hole)


ACTIVE = HandRanks.load()
CHEN = HandRanks()
