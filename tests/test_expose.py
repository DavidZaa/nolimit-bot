"""bot/expose.py: hide strength when strong, look scary when weak."""

import random
from time import perf_counter_ns

import pytest

from bot.discard import default_rules
from bot.expose import choose_expose
from bot.state import cards_from_str as C

RULES = default_rules()


def shown(kept, board):
    return choose_expose(C(kept), C(board), RULES)


@pytest.mark.parametrize("kept, board, show", [
    ("AsKd", "Kh7c2s", "As"),  # top pair of kings: show the ace, not the king that pairs
    ("Kd7d", "Kh7c2s", "7d"),  # two pair: both pair the board; show the lower one
    ("Qc9c", "9h9d4s", "Qc"),  # trips with the nine: show the queen
    ("Ah5h", "Kh9h2h", "5h"),  # made flush: both are hearts; show the smaller
    ("8c6d", "7h5s4d", "6d"),  # straight: both cards are in the run, so they tie; show the lower
])
def test_strong_hands_show_the_card_that_contributes_least(kept, board, show):
    assert shown(kept, board) == C(show)[0]


@pytest.mark.parametrize("kept, board, show", [
    ("Ac5d", "Kh7s2s", "Ac"),  # nothing: the ace is the scariest card
    ("Jd4s", "Kh7s2s", "4s"),  # no ace: the board-suit card (a spade) threatens the flush
    ("Qc4d", "Kh7s2c", "Qc"),  # no ace, no board suit with two: the highest card
    ("Ah5h", "Kh9h2c", "Ah"),  # a flush draw is a weak hand: show the ace
    ("9c8d", "Th7s2d", "9c"),  # open-ender: no ace, no suit twice on board: the higher card
])
def test_weak_hands_and_draws_show_the_scariest_card(kept, board, show):
    assert shown(kept, board) == C(show)[0]


def test_always_one_of_the_kept_cards_and_fast():
    rng = random.Random(6)
    deck = RULES.deck()
    spots = [rng.sample(deck, 5) for _ in range(2000)]
    times = []
    for c in spots:
        t = perf_counter_ns()
        out = choose_expose(c[:2], c[2:], RULES)
        times.append(perf_counter_ns() - t)
        assert out in c[:2]
    times.sort()
    assert times[len(times) // 2] < 100_000
