"""What an opponent's exposed card says about its hand, with the board. Bay Area only.

After the reveal each opponent has one public card. With the board it already
makes a visible hand: strong_visible() is true when that card pairs the board or
makes two pair or better with it (trips on a paired board, two pair with a
paired board, or any two-pair-plus category once four board cards are out).
Nothing is read into a low card that misses: at showdown the field exposes its
lower card about twice as often as its higher, so a low card is not weakness.
"""

from __future__ import annotations

from bot.handeval import category, hand_value
from bot.rules import TWO_PAIR


def strong_visible(card: int, board, rules) -> bool:
    if card < 0 or len(board) < 3:
        return False
    rank = card >> 2
    board_ranks = [c >> 2 for c in board]
    if rank in board_ranks:
        return True  # pairs the board (or better)
    if len(board) >= 4:  # four or more cards plus one: a legal five exists
        try:
            return category(hand_value((card,), board, rules), rules) >= TWO_PAIR
        except (ValueError, KeyError):
            return False
    return False


def strongest_visible_seats(state) -> list[int]:
    """Opponents still in whose exposed card is strong with this board."""
    exposed = getattr(state, "exposed", ()) or ()
    out = []
    for s, card in enumerate(exposed):
        if s != state.seat and s < len(state.in_hand) and state.in_hand[s] and card >= 0:
            if strong_visible(card, state.board, state.rules):
                out.append(s)
    return out
