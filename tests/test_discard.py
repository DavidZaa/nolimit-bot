"""bot/discard.py: keep the best two of three."""

import random
from time import perf_counter_ns

import pytest

from bot.discard import choose_discard, default_rules, keep_score
from bot.state import cards_from_str as C

RULES = default_rules()


def discard(hole, board):
    return choose_discard(C(hole), C(board), RULES)


@pytest.mark.parametrize("hole, board, thrown", [
    ("Kh7c2d", "Kd9s4c", "2d"),  # keeps the made pair (K), throws the deuce over the seven: better kicker
    ("AcQd8h", "8s5d2c", "Qd"),  # pair of eights beats ace-high: the queen goes, the ace stays as kicker
    ("9s9h4c", "9d7c2h", "4c"),  # keeps the set
    ("AhKc6h", "Jh8h3s", "Kc"),  # keeps the flush draw (A6h) over unpaired high cards (AK)
    ("9c8dKs", "Th7s2d", "Ks"),  # keeps the open-ender over a lone overcard
    ("AsKdQh", "7c5d2s", "Qh"),  # nothing made, no draws: keeps the two biggest overcards
    ("7c7dAs", "Kh9s2c", "As"),  # a pocket pair is a made pair; an ace alone isn't
])
def test_keeps_the_best_two(hole, board, thrown):
    assert discard(hole, board) == C(thrown)[0]


def test_made_hands_come_before_draws():
    # A weak pair (sixes) still beats a naked flush draw with overcards.
    assert discard("6c AhKh".replace(" ", ""), "6dJh4h") in C("AhKh")


def test_before_the_flop_it_keeps_the_best_starting_hand():
    assert discard("AsAd7c", "") == C("7c")[0]
    assert discard("KsQs2d", "") == C("2d")[0]


def test_the_choice_is_one_of_our_cards_and_deterministic():
    rng = random.Random(4)
    deck = RULES.deck()
    for _ in range(500):
        cards = rng.sample(deck, 6)
        hole, board = cards[:3], cards[3:]
        out = choose_discard(hole, board, RULES)
        assert out in hole and choose_discard(hole, board, RULES) == out
        kept = [c for c in hole if c != out]
        assert keep_score(kept, board, RULES) >= max(
            keep_score([c for c in hole if c != other], board, RULES) for other in hole)


def test_it_runs_in_under_100_microseconds():
    rng = random.Random(5)
    deck = RULES.deck()
    spots = [rng.sample(deck, 6) for _ in range(2000)]
    for c in spots[:50]:
        choose_discard(c[:3], c[3:], RULES)  # warm caches
    times = []
    for c in spots:
        t = perf_counter_ns()
        choose_discard(c[:3], c[3:], RULES)
        times.append(perf_counter_ns() - t)
    times.sort()
    assert times[len(times) // 2] < 100_000 and times[int(len(times) * 0.99)] < 100_000, (
        times[len(times) // 2], times[int(len(times) * 0.99)])
