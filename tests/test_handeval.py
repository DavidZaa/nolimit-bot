import itertools
import random

import pytest

from bot.state import cards_from_str as C
from engine.handeval import best_hand, category, evaluate5
from engine.rules import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    NLHE,
    PAIR,
    PLO,
    QUADS,
    SHORT_DECK,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
)

LADDER = [
    ("AsKd9h7c2s", HIGH_CARD),
    ("AsAd9h7c2s", PAIR),
    ("AsAd9h9c2s", TWO_PAIR),
    ("AsAdAh7c2s", TRIPS),
    ("5s4d3h2cAs", STRAIGHT),
    ("As9s7s4s2s", FLUSH),
    ("AsAdAh2c2s", FULL_HOUSE),
    ("AsAdAhAc2s", QUADS),
    ("5s4s3s2sAs", STRAIGHT_FLUSH),
]


def score(s, variant=NLHE):
    return evaluate5(C(s), variant)


def test_each_category_is_recognised():
    for cards, cat in LADDER:
        assert category(score(cards), NLHE) == cat, cards


def test_categories_rank_in_standard_order():
    scores = [score(cards) for cards, _ in LADDER]
    assert scores == sorted(scores)


@pytest.mark.parametrize(
    "better, worse",
    [
        ("AsAdKh7c2s", "AsAdQhJcTs"),  # pair kicker
        ("KsKdQhQc2s", "KsKdJhJcAs"),  # second pair outranks the kicker
        ("6s5d4h3c2s", "5s4d3h2cAs"),  # six-high straight beats the wheel
        ("AsKsQsJsTs", "KsQsJsTs9s"),
        ("3s3d3h2c2s", "2s2d2hAcAs"),  # full house compares trips first
        ("AsKs9s7s3s", "AsKs9s7s2s"),  # flush fifth card
    ],
)
def test_kickers_and_tiebreaks(better, worse):
    assert score(better) > score(worse)


def test_identical_strength_ties():
    assert score("AsKd9h7c2s") == score("AdKh9c7s2d")


def test_best_hand_picks_best_five_of_seven():
    # Board makes a straight; the pocket pair is irrelevant.
    s = best_hand(C("2c2d"), C("9s8dTh7c6h"), NLHE)
    assert category(s, NLHE) == STRAIGHT
    assert s == score("Ts9s8d7c6h")


def test_best_hand_matches_brute_force_on_random_hands():
    rng = random.Random(7)
    deck = list(range(52))
    for _ in range(300):
        cards = rng.sample(deck, 7)
        brute = max(evaluate5(combo, NLHE) for combo in itertools.combinations(cards, 5))
        assert best_hand(cards[:2], cards[2:], NLHE) == brute


def test_short_deck_wheel_is_ace_six_through_nine():
    s = score("As6d7h8c9s", SHORT_DECK)
    assert category(s, SHORT_DECK) == STRAIGHT
    assert s < score("6s7d8h9cTs", SHORT_DECK)


def test_short_deck_flush_beats_full_house_and_trips_beat_straight():
    assert score("As9s7s8sJs", SHORT_DECK) > score("AsAdAh6c6s", SHORT_DECK)
    assert score("6s6d6hAcKs", SHORT_DECK) > score("TsJdQhKcAs", SHORT_DECK)


def test_plo_must_use_exactly_two_hole_cards():
    # Four spades on board plus one spade in hand is not a flush in Omaha.
    hole = C("AsKdQd2c")
    board = C("9s7s5s3sJh")
    assert category(best_hand(hole, board, PLO), PLO) != FLUSH
    # The same cards in hold'em-style "any five" would be a flush.
    assert category(best_hand(hole, board, NLHE), NLHE) == FLUSH


# --- Cross-check against PokerKit's hand rankings -------------------------

pokerkit = pytest.importorskip("pokerkit")

from dataclasses import replace  # noqa: E402

from bot.state import card_to_str  # noqa: E402
from engine.rules import STANDARD_ORDER  # noqa: E402

# PokerKit's short deck keeps straight above trips; mirror that for the comparison.
_SD_POKERKIT = replace(
    SHORT_DECK,
    category_order=(HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH),
)


def _s(cards):
    return "".join(card_to_str(c) for c in cards)


@pytest.mark.parametrize(
    "variant, pk_hand",
    [
        (NLHE, pokerkit.StandardHighHand),
        (_SD_POKERKIT, pokerkit.ShortDeckHoldemHand),
        (PLO, pokerkit.OmahaHoldemHand),
    ],
    ids=["nlhe", "short_deck", "plo"],
)
def test_ordering_matches_pokerkit(variant, pk_hand):
    rng = random.Random(11)
    deck = variant.deck()
    n_hole = variant.hole_cards
    for _ in range(1500):
        cards = rng.sample(deck, 5 + 2 * n_hole)
        board, h1, h2 = cards[:5], cards[5 : 5 + n_hole], cards[5 + n_hole :]
        ours = best_hand(h1, board, variant) - best_hand(h2, board, variant)
        a = pk_hand.from_game(_s(h1), _s(board))
        b = pk_hand.from_game(_s(h2), _s(board))
        theirs = (a > b) - (a < b)
        assert (ours > 0) - (ours < 0) == theirs, (_s(h1), _s(h2), _s(board))


def test_standard_order_is_the_default():
    assert NLHE.category_order == STANDARD_ORDER
