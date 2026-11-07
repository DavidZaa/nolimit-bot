"""The Bay Area selection, in the kit's terms: which original card to discard, which to show.

    choose_select(hole, board, rules=None) -> (discard_index, reveal_index)

Indexes are into the three hole cards in dealing order, as the kit's
Action("select", cards=(discard, reveal)) wants them. The discard keeps the best
two cards (bot/discard.py); the reveal hides a strong hand's strength or makes a
weak one look scary (bot/expose.py).
"""

from __future__ import annotations

from bot.discard import choose_discard
from bot.expose import choose_expose

DEFAULT_SELECT = (2, 0)  # the kit's own fallback: discard the third card, show the first


def choose_select(hole, board, rules=None) -> tuple[int, int]:
    hole = list(hole)
    if len(hole) != 3 or len(set(hole)) != 3:
        return DEFAULT_SELECT
    discard = choose_discard(hole, board, rules)
    kept = [c for c in hole if c != discard]
    reveal = choose_expose(kept, board, rules)
    return hole.index(discard), hole.index(reveal)
